# Parts of the suite on four threads and with `--check-bounds=yes`,
# side by side on one node (`test/some_tests.jl`), for checking one part
# without the whole suite (`suite.toml`). Each part is a pair of runs,
# `<part>-t4` and `<part>-cb`, submitted together, one `--run` each:
#
#     simlaunch snapshot . --study twoldg-<commit> --rev <commit> --note "<what it is for>"
#     simlaunch prepare twoldg-<commit> --site symmetry --spec jobs/tests_by_part.jl
#     simlaunch submit twoldg-<commit> jobs/tests_by_part.jl --site symmetry \
#         --run particles-t4 --run particles-cb
#
# A new part is a new entry in `parts` and a new snapshot.
parts = [("particles", "predictor_tests.jl dynamic_tests.jl diagnostics_tests.jl"),
         ("flux", "extraction_tests.jl prescribed_tests.jl")]
args = ["--depwarn=yes", "--warn-overwrite=yes"]
runs = [(label = "$part-$mode", cmd = "test/some_tests.jl $files", threads = 4,
         julia_args = mode == "cb" ? [args; "--check-bounds=yes"] : args)
        for (part, files) in parts for mode in ("t4", "cb")]

(name = "tests_by_part",
 description = "TreeWaveOLDG: parts of the test suite, four threads and bounds checked",
 env = "test",
 mode = "parallel",
 julia_args = args,
 site = (symmetry = (time = "1:00:00",),),
 run = runs)
