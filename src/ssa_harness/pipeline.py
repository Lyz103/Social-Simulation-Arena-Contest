"""One auditable path from a request to an official response; never submits externally."""

from dataclasses import asdict, dataclass, replace
import time

from .archive import RunArchive
from .context import ContextBuilder
from .contracts import ForecastRequest, response_envelope
from .ensemble import EnsembleEngine, ForecastMember
from .history import normalized_request
from .llm import ModelError
from .prompt import ForecastPromptCompiler, PredictionValidator, InvalidModelOutput
from .ranking import CandidateBuilder
from .retrieval.utils import digest, parse_instant, utcnow


@dataclass(frozen=True)
class PipelineConfig:
    entrant: str = "local-experiment"
    mode: str = "snapshot"
    research_policy: str = "required"
    allow_baseline_fallback: bool = False
    repair_attempts: int = 1
    max_elapsed_seconds: int = 480
    sd_scale: float = 1.0

    def __post_init__(self):
        import math
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.entrant):
            raise ValueError("invalid entrant")
        if self.mode not in ("snapshot", "live") or self.research_policy not in (
            "required",
            "disabled",
        ):
            raise ValueError("invalid pipeline policy")
        if (
            self.repair_attempts not in (0, 1)
            or not 1 <= self.max_elapsed_seconds <= 540
        ):
            raise ValueError("invalid pipeline budget")
        if not math.isfinite(self.sd_scale) or self.sd_scale <= 0:
            raise ValueError("invalid sd calibration scale")


class ForecastPipeline:
    VERSION = "pipeline-v1"

    def __init__(
        self,
        *,
        archive: RunArchive,
        clients,
        research_engine=None,
        context_builder=None,
        compiler=None,
        ensemble=None,
        config=None,
        clock=utcnow,
        monotonic=time.monotonic,
        source_packages=(),
    ):
        self.archive, self.clients, self.research_engine = (
            archive,
            dict(clients),
            research_engine,
        )
        self.context_builder = context_builder or ContextBuilder()
        self.compiler = compiler or ForecastPromptCompiler()
        self.ensemble = ensemble or EnsembleEngine(failure_policy="drop_invalid")
        self.config = config or PipelineConfig()
        self.clock, self.monotonic = clock, monotonic
        self.source_packages = json_copy(list(source_packages))
        if self.config.mode == "live" and any(
            getattr(c, "synthetic", False) for c in self.clients.values()
        ):
            raise ValueError("live pipeline cannot use fixture models")
        if not self.clients and not self.config.allow_baseline_fallback:
            raise ValueError(
                "configure a model or explicitly allow baseline-only operation"
            )
        if self.config.research_policy == "required" and self.research_engine is None:
            raise ValueError("research engine required")

    def identity(self):
        from .retrieval.engine import PIPELINE_VERSION

        research = self.research_engine
        return {
            "version": self.VERSION,
            "config": asdict(self.config),
            "clients": {name: c.identity() for name, c in self.clients.items()},
            "context": {
                "version": self.context_builder.VERSION,
                **asdict(self.context_builder.config),
            },
            "compiler": {
                "version": self.compiler.VERSION,
                "system_message": self.compiler.system_message,
                "params": self.compiler.model_params,
            },
            "ensemble": {
                "version": self.ensemble.VERSION,
                "weights": self.ensemble.configured_weights,
                "min_sd": self.ensemble.min_sd,
                "max_sd": self.ensemble.max_sd,
                "failure_policy": self.ensemble.failure_policy,
                "baseline_window": self.ensemble.baseline_factory.recent_window,
                "baseline_min_sd": self.ensemble.baseline_factory.min_sd,
            },
            "research": (
                None
                if research is None
                else {
                    "version": PIPELINE_VERSION,
                    "retriever": research.retriever.identity(),
                    "planner": research.planner.identity(),
                    "config": asdict(research.config),
                }
            ),
            "candidate_builder": CandidateBuilder.VERSION,
            "source_packages_hash": digest(self.source_packages),
        }

    def run_envelope(self, envelope, *, asof=None, metadata=None):
        request = ForecastRequest.from_envelope(envelope, asof=asof, metadata=metadata)
        self.archive.bind_request(
            self.config.entrant, request.request_id, digest(envelope)
        )
        return self.run(request)

    def run(self, request):
        # Snapshot mutable dictionaries before computing identities.
        request = ForecastRequest.from_dict(request.to_dict())
        entrant = self.config.entrant
        input_hash, config_hash = request.request_hash, digest(self.identity())
        existing = self.archive.load(entrant, request.round_id, input_hash, config_hash)
        if existing is not None:
            return existing
        with self.archive.claim(
            request.round_id, entrant, self.config.max_elapsed_seconds + 60
        ):
            existing = self.archive.begin(
                entrant, request.round_id, input_hash, config_hash
            )
            if existing is not None:
                return existing
            start = self.monotonic()

            def remaining():
                seconds = self.config.max_elapsed_seconds - (self.monotonic() - start)
                if self.config.mode == "live":
                    seconds = min(
                        seconds,
                        (
                            parse_instant(request.deadline) - self.clock()
                        ).total_seconds(),
                    )
                if seconds <= 0:
                    raise TimeoutError("pipeline deadline/time budget exceeded")
                return seconds

            def event(stage, value):
                self.archive.event(entrant, request.round_id, stage, value)

            try:
                if parse_instant(request.asof) > self.clock():
                    raise ValueError("information cutoff is in the future")
                remaining()
                if self.source_packages:
                    from .sources import merge_source_history

                    history_request = merge_source_history(
                        request, self.source_packages
                    )
                else:
                    history_request = request
                clean, history = normalized_request(history_request)
                event(
                    "request",
                    {
                        "original": request.to_dict(),
                        "normalized": clean.to_dict(),
                        "history_warnings": history.warnings,
                    },
                )
                research = None
                if self.config.research_policy == "required":
                    research = self.research_engine.retrieve(
                        clean.to_research_request(mode=self.config.mode),
                        entrant=entrant,
                    )
                    if (
                        self.config.mode == "live"
                        and research.temporal_assurance == "synthetic_demo"
                    ):
                        raise ValueError(
                            "synthetic evidence forbidden in live operation"
                        )
                    event("research", research.to_dict())
                remaining()
                pool = None
                if clean.target_type == "ranking_list":
                    pool = CandidateBuilder().build(
                        clean, research.documents if research else ()
                    )
                    # Candidate items are suggestions; open output remains open.
                    clean = replace(clean, candidates=pool.items)
                    event("candidates", asdict(pool))
                context = self.context_builder.build(clean, research)
                event("context", asdict(context))
                compiled = self.compiler.compile(clean, context)
                event("prompt", asdict(compiled))
                members, failures, model_runs = [], [], []
                previous_events = self.archive.events(entrant, request.round_id)
                for name, client in self.clients.items():
                    cache_key = digest(
                        {
                            "member": name,
                            "client": client.identity(),
                            "prompt": compiled.prompt_hash,
                        }
                    )
                    cached = next(
                        (
                            e["data"]
                            for e in reversed(previous_events)
                            if e["stage"] == "model_validated"
                            and e["data"]["cache_key"] == cache_key
                        ),
                        None,
                    )
                    if cached:
                        members.append(
                            ForecastMember(name, "llm", cached["prediction"])
                        )
                        model_runs.append(cached)
                        continue
                    repair = None
                    for attempt in range(self.config.repair_attempts + 1):
                        remaining()
                        try:
                            reply = client.generate(
                                compiled, timeout=remaining(), repair=repair
                            )
                            raw = {
                                "member_id": name,
                                "cache_key": cache_key,
                                "attempt": attempt,
                                **asdict(reply),
                            }
                            event("model_raw", raw)
                            model_runs.append(raw)
                            remaining()
                            pred = PredictionValidator().parse(reply.text, clean)
                            valid = {**raw, "prediction": pred}
                            event("model_validated", valid)
                            members.append(
                                ForecastMember(
                                    name,
                                    "llm",
                                    pred,
                                    metadata={
                                        "model": reply.model,
                                        "prompt_hash": compiled.prompt_hash,
                                    },
                                )
                            )
                            break
                        except InvalidModelOutput as exc:
                            event(
                                "model_invalid",
                                {
                                    "member_id": name,
                                    "attempt": attempt,
                                    "error": str(exc),
                                },
                            )
                            if attempt < self.config.repair_attempts:
                                repair = {"raw": reply.text, "error": str(exc)}
                                continue
                            failures.append({"member_id": name, "error": str(exc)})
                        except ModelError as exc:
                            event(
                                "model_failed",
                                {
                                    "member_id": name,
                                    "attempt": attempt,
                                    "error": str(exc),
                                },
                            )
                            failures.append({"member_id": name, "error": str(exc)})
                        break
                if not members and not self.config.allow_baseline_fallback:
                    raise ModelError("all models failed; baseline fallback is disabled")
                remaining()
                final = self.ensemble.combine(clean, members)
                if failures:
                    final = replace(
                        final,
                        degraded=True,
                        warnings=final.warnings
                        + [f"model failed: {f['member_id']}" for f in failures],
                    )
                prediction = final.prediction
                if self.config.sd_scale != 1 and clean.target_type != "ranking_list":
                    prediction = json_copy(prediction)
                    normals = (
                        prediction["cells"].values()
                        if clean.target_type == "profile_energy"
                        else [prediction]
                    )
                    for normal in normals:
                        normal["sd"] *= self.config.sd_scale
                    final = replace(
                        final,
                        prediction=prediction,
                        warnings=final.warnings
                        + ["sd scale applied: " + str(self.config.sd_scale)],
                    )
                response = response_envelope(clean, final.prediction)
                result = {
                    "schema_version": "ssa-harness-run-v1",
                    "request": request.to_dict(),
                    "normalized_request": clean.to_dict(),
                    "input_hash": input_hash,
                    "pipeline_config_hash": config_hash,
                    "pipeline": self.identity(),
                    "research": research.to_dict() if research else None,
                    "context": asdict(context),
                    "compiled_prompt": asdict(compiled),
                    "candidate_pool": asdict(pool) if pool else None,
                    "model_runs": model_runs,
                    "member_failures": failures,
                    "final": asdict(final),
                    "response": response,
                    "elapsed_seconds": self.monotonic() - start,
                }
                remaining()
                event("final", {"response": response, "degraded": final.degraded})
                self.archive.save(
                    entrant, request.round_id, input_hash, config_hash, result
                )
                return self.archive.load(
                    entrant, request.round_id, input_hash, config_hash
                )
            except Exception as exc:
                event("failed", {"error_type": type(exc).__name__})
                raise


def json_copy(value):
    import json

    return json.loads(json.dumps(value))
