#!/bin/bash

cd ./built/bin/
for i in `ls test_*`
  do ./$i
done