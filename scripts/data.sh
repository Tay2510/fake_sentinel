#!/usr/bin/env bash

TARGET_DIR=$1

mkdir "$TARGET_DIR"

for i in {0..9}
  do
    mv "dfdc_train_part_0$i.zip" "dfdc_train_part_$i.zip"
  done

for i in {0..49}
  do
    name="dfdc_train_part_$i"
    zip_file="$name.zip"
    unzip "$zip_file" && rm "$zip_file"
    mv "$name/*.mp4" "$TARGET_DIR" && rm -rf "$name"
  done
