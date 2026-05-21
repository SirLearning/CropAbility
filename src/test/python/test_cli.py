"""Tests for ``cropability.cli.main`` argument parsing."""

import pytest

from cropability.cli.main import build_parser


class TestCliParser:
    def test_call_variants_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            [
                "call-variants",
                "-r",
                "ref.fa",
                "-b",
                "a.bam",
                "-o",
                "out.vcf",
                "--mode",
                "hybrid",
                "--dry-run",
            ]
        )
        assert ns.command == "call-variants"
        assert ns.mode == "hybrid"
        assert ns.dry_run is True

    def test_pileup_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            [
                "pileup",
                "-r",
                "ref.fa",
                "-b",
                "a.bam",
                "-o",
                "out.tsv",
            ]
        )
        assert ns.command == "pileup"
        assert ns.reference == "ref.fa"

    def test_export_subcommand_removed(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "-o", "model.pt"])
