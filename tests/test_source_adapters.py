import io
import json
import zipfile

from ssa_harness.source_adapters import (
    CiviqsRetriever,
    YouGovRetriever,
    civiqs_package,
    parse_civiqs_html,
    parse_yougov_xlsx,
    yougov_package,
)


def _civiqs_html():
    payload = {
        "run_id": "run-1",
        "end_date": "2026-09-24",
        "job_description": {
            "name": "approve_president_trump_2025",
            "display_text": "Trump Approval",
            "population_model": "registered voters",
            "display_net": {"minuend": "Approve", "subtrahend": "Disapprove"},
        },
        "topline": {
            "line_chart_data": [
                {"key": "Approve", "values": [{"date": 1790208000000, "value": 0.42}]},
                {
                    "key": "Disapprove",
                    "values": [{"date": 1790208000000, "value": 0.51}],
                },
            ]
        },
    }
    return (
        '<html><script>"routes/_app.results_.$question":'
        + json.dumps(payload)
        + "</script></html>"
    ).encode()


def _yougov_xlsx():
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    prels = "http://schemas.openxmlformats.org/package/2006/relationships"
    names = ["US Registered Voters", "Democrat"]

    def cell(v, string=False):
        return (
            f'<c t="inlineStr"><is><t>{v}</t></is></c>'
            if string
            else f"<c><v>{v}</v></c>"
        )

    def sheet():
        rows = [
            ("Question", "2026-09-21", True),
            ("Approve", 0.42, False),
            ("Disapprove", 0.5, False),
            ("Not sure", 0.08, False),
            ("Unweighted base", 100, False),
            ("Base", 200, False),
        ]
        return (
            f'<worksheet xmlns="{main}"><sheetData>'
            + "".join(
                "<row>" + cell(a, True) + cell(b, c) + "</row>" for a, b, c in rows
            )
            + "</sheetData></worksheet>"
        ).encode()

    wb = (
        f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>'
        + "".join(
            f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>'
            for i, n in enumerate(names, 1)
        )
        + "</sheets></workbook>"
    )
    rs = (
        f'<Relationships xmlns="{prels}">'
        + "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, 3)
        )
        + "</Relationships>"
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rs)
        z.writestr("xl/worksheets/sheet1.xml", sheet())
        z.writestr("xl/worksheets/sheet2.xml", sheet())
    return out.getvalue()


def test_civiqs_parser_preserves_capture_and_computes_net():
    raw = _civiqs_html()
    assert parse_civiqs_html(raw)["run_id"] == "run-1"
    package = civiqs_package(
        raw,
        tracker="approve_president_trump_2025",
        url="https://civiqs.com/results/approve_president_trump_2025",
        captured_at="2026-09-25T00:00:00Z",
    )
    assert package["history"][0]["value"] == -9.0
    assert package["manifest"]["content_hash"].startswith("sha256:")


def test_yougov_parser_reads_xlsx_relationships_and_points():
    waves = parse_yougov_xlsx(_yougov_xlsx(), scored_cells=["Democrat"])
    assert waves[0]["cells"]["Democrat"]["approve"] == 42.0
    package = yougov_package(
        _yougov_xlsx(),
        tracker="donald-trump-approval",
        url="https://api-test.yougov.com/public-data/v5/us/trackers/donald-trump-approval/download/",
        captured_at="2026-09-25T00:00:00Z",
        scored_cells=["Democrat"],
    )
    assert package["history"][0]["value"]["US Registered Voters"] == 42.0


def test_retriever_identities_are_keyless():
    assert CiviqsRetriever().identity()["api_key"] is False
    assert YouGovRetriever().identity()["api_key"] is False
