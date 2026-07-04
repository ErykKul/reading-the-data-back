# P2: reused + extended rdm-integration DDI-CDI generator

`cdi_generator_ext.py` = fork of libis/rdm-integration `image/cdi_generator_jsonld.py` (manifest-driven,
`{dataset_pid, files:[{csv_path,...}]}` -> DDI-CDI 1.0 JSON-LD), EXTENDED so each variable's
SubstantiveValueDomain carries a `distributionKind` (binary/ordinal/count/proportion/heavy-tailed/continuous)
derived from streaming stats. The stock generator only emits XSD Integer/Double, which cannot distinguish
count vs ordinal vs binary, or proportion vs continuous -- the facet the model-data mismatch operator needs.
Deps: chardet, datasketch, python-dateutil. Run: `python cdi_generator_ext.py --manifest m.json --output out.jsonld`.
Verified on a smoke CSV: count/proportion/binary/heavy-tailed/ordinal all correct.
