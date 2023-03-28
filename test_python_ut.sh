PYTHON=python
for file in $(ls -ld $(find ./test/python))
do
  if [[ $file == */test_*.py ]]
  then
    echo $file
    ${PYTHON} $file
    sleep 1s
  fi
done