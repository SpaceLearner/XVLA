#!/bin/bash
for i in {0..23}
do
    echo $i
    python generate_bboxes.py --id $i --gpu 0
done