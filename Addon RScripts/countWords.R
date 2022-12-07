
file <- "/Volumes/Jellyfish/Dropbox/Manuscripts/_ Under Revision/Artificial intelligence maps giant kelp forests from satellite imagery/Annotations Data/Final Data/test/via_region_data.json"
readLines(file)

library(stringr)
str_count(readLines(file), "kelp")

# Test 537
# Train: 2368
# Val: 440