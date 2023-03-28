# Documents
## Generate documents

1. install requirements

```
cd docs
pip install -r requirements.txt
```

2. generate htmls

```
# make clean
make html
```

## Update API Reference

If a new module added or an existed module deleted, then update
the .rst files in docs/apis

```
sphinx-apidoc -o ./apis ../graphlearn_torch/python/{module_name}
# update docs/apis/modules
make clean
make html
```