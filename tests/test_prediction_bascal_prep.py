import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orca.scripts.prediction_bascal_prep import index_pileup


def pileup_line(contig, position=1, ending=b"\n"):
    return (
        f"{contig}\t{position}\tA\t10\t..........\tIIIIIIIIII".encode("utf-8")
        + ending
    )


class TestIndexPileup(unittest.TestCase):
    def assert_index(self, groups):
        with TemporaryDirectory() as directory:
            pileup = Path(directory) / "input.pileup"
            index = Path(directory) / "input.pileup.index"
            data = b"".join(chunk for _, chunk in groups)
            pileup.write_bytes(data)
            index.write_text("stale index\n", encoding="utf-8")

            index_pileup(pileup, index)

            with index.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["id", "start", "end"])
                rows = list(reader)
            self.assertEqual([row["id"] for row in rows], [name for name, _ in groups])
            offset = 0
            for row, (name, chunk) in zip(rows, groups):
                start, end = int(row["start"]), int(row["end"])
                self.assertEqual((start, end), (offset, offset + len(chunk)))
                self.assertEqual(data[start:end], chunk)
                self.assertEqual(
                    {line.split(b"\t", 1)[0].decode("utf-8")
                     for line in data[start:end].splitlines()},
                    {name},
                )
                offset = end
            self.assertEqual(offset, len(data))

    def test_empty_pileup_overwrites_stale_index(self):
        self.assert_index([])

    def test_single_line_contig(self):
        self.assert_index([("only", pileup_line("only"))])

    def test_single_contig_multiple_positions(self):
        self.assert_index([("only", pileup_line("only") + pileup_line("only", 2))])

    def test_multiple_contigs_including_single_line_last_contig(self):
        self.assert_index([
            ("first", pileup_line("first") + pileup_line("first", 12)),
            ("middle_long_name", pileup_line("middle_long_name")),
            ("last", pileup_line("last", 123)),
        ])

    def test_multiple_positions_in_last_contig(self):
        self.assert_index([
            ("first", pileup_line("first")),
            ("last", pileup_line("last") + pileup_line("last", 2)),
        ])

    def test_no_trailing_newline(self):
        self.assert_index([
            ("first", pileup_line("first")),
            ("last", pileup_line("last", ending=b"")),
        ])

    def test_single_line_without_trailing_newline(self):
        self.assert_index([("only", pileup_line("only", ending=b""))])

    def test_crlf_byte_ranges(self):
        self.assert_index([
            ("first", pileup_line("first", ending=b"\r\n")),
            ("last", pileup_line("last", ending=b"\r\n")),
        ])

    def test_utf8_contig_byte_ranges(self):
        self.assert_index([
            ("transcript_α", pileup_line("transcript_α")),
            ("last", pileup_line("last")),
        ])


class TestBasecallingPipeline(unittest.TestCase):
    def run_cli(self, module, *args):
        result = subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def extract(self, directory, data, n_processes):
        pileup = Path(directory) / "input.pileup"
        pileup.write_bytes(data)
        self.run_cli(
            "orca.scripts.prediction_bascal_prep",
            "--pileup", pileup, "--work_dir", directory,
            "--prefix", "example", "--n_processes", n_processes,
        )

    def test_empty_pileup_cli(self):
        with TemporaryDirectory() as directory:
            self.extract(directory, b"", 1)
            for suffix in ["bascal.feature.per.site", "bascal.feature.index"]:
                with (Path(directory) / f"example.{suffix}").open(newline="") as handle:
                    self.assertEqual(list(csv.DictReader(handle)), [])

    def assert_extraction_and_merge(self, contigs, n_processes):
        with TemporaryDirectory() as directory:
            # Five consecutive sites yield one merged feature window per contig.
            lines = [pileup_line(contig, pos) for contig in contigs for pos in range(1, 6)]
            self.extract(directory, b"".join(lines).rstrip(b"\n"), n_processes)
            base_path = Path(directory) / "example.bascal.feature.per.site"
            with base_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                sorted((r["id"], int(r["position"])) for r in rows),
                sorted((contig, pos) for contig in contigs for pos in range(5)),
            )
            for row in rows:
                self.assertEqual(float(row["depth"]), 10)
                self.assertEqual(float(row["Qual_Mean"]), 40)
                self.assertEqual(float(row["Mismatch_Ratio"]), 0)
            with (Path(directory) / "example.bascal.feature.index").open(newline="") as handle:
                self.assertEqual(
                    {r["id"] for r in csv.DictReader(handle)}, set(contigs)
                )

            # Synthetic signal features isolate the indexer's downstream effect.
            signal_header = ["id", "position", "kmer"] + [f"{i}_shape" for i in range(50)]
            signal_data = (",".join(signal_header) + "\n").encode("ascii")
            signal_index = ["id,start,end\n"]
            for contig in contigs:
                chunk = "".join(
                    ",".join([contig, str(pos), "AAAAA"] + ["1.0"] * 50) + "\n"
                    for pos in range(5)
                ).encode("ascii")
                start = len(signal_data)
                signal_data += chunk
                signal_index.append(f"{contig},{start},{len(signal_data)}\n")
            (Path(directory) / "example.signal.feature.per.site").write_bytes(signal_data)
            (Path(directory) / "example.signal.feature.index").write_bytes(
                "".join(signal_index).encode("ascii")
            )
            self.run_cli(
                "orca.scripts.prediction_feature_merge",
                "--work_dir", directory, "--prefix", "example",
                "--n_processes", n_processes,
            )
            with (Path(directory) / "example.merged.feature.per.site").open(newline="") as handle:
                merged = list(csv.DictReader(handle))
            self.assertEqual(
                sorted((r["id"], int(r["position"])) for r in merged),
                sorted((contig, 2) for contig in contigs),
            )
            for row in merged:
                self.assertEqual(row["kmer"], "AAAAA")
                self.assertEqual(float(row["depth"]), 10)
                self.assertEqual(len(row), 284)
                self.assertNotIn(None, row)

    def test_single_contig_survives_extraction_and_merge(self):
        self.assert_extraction_and_merge(["only"], 1)

    def test_final_contig_survives_parallel_extraction_and_merge(self):
        self.assert_extraction_and_merge(["first", "middle", "last"], 2)


if __name__ == "__main__":
    unittest.main()
