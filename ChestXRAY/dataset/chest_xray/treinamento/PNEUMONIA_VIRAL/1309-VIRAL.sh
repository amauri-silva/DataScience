#!/bin/bash


rename_test3 () { 
    local file_spec=$*
    local counter=0
    local suffix=${SUFFIX:-VIRAL}
    for file in $(ls -1 $file_spec) ; do
        echo $file
        extension=${file#*.}
        if [[ "$extension" == "$file" ]]; then
            extension=""
        else
            extension=".$extension"
        fi
        mv -v $file "$(printf "%01d" $counter)-${suffix}${extension}"
        let $((counter++))
    done
}
rename_test3
