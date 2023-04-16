#!/bin/bash
# set -e


# this script converts a list of jupyter_notebooks.ipynb to pdf

OUTPUT_DIR="pdfy"

#CONVERT_OPTION include ['PDFviaHTML', 'asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'pdfviahtml', 'python', 'rst', 'script', 'slides', 'webpdf']
CONVERT_OPTION="PDFviaHTML" # simple pdf is not enough Oo


NOTEBOOK_PATHS=("10_intro_setup" 
     "20_vector_add"
     "30_matrix_matrix_multiplication"
     "40_parallel_reduction"
     "50_thrust")

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

for i in ${!NOTEBOOK_PATHS[@]};
do
    NOTEBOOK=${NOTEBOOK_PATHS[$i]}
    echo "parsing notebook[$i]: ${NOTEBOOK}.ipynb"
    jupyter-nbconvert --to ${CONVERT_OPTION} ${NOTEBOOK}.ipynb 
    mv ${NOTEBOOK}.pdf ${OUTPUT_DIR}/
done

