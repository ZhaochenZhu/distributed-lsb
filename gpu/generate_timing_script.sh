#!/bin/bash

# Usage: ./generate_timing_script.sh <suffix>
# Example: ./generate_timing_script.sh ppsum

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <suffix>"
    exit 1
fi

SUFFIX="$1"
SCRIPT="run_timing_${SUFFIX}.sh"

cat > "$SCRIPT" <<EOF
#!/bin/bash

# Output file
OUTFILE="timing_results_${SUFFIX}.txt"

# Clear previous output
echo "" > \$OUTFILE

# Problem sizes: 10^3 to 10^8
for exp in {3..8}
do
    n=\$((10**exp))
    echo "Running with n=\$n"
    
    # Run the ${SUFFIX} executable, capture output
    ./radix_sort_${SUFFIX} \$n > temp_output.txt

    # Extract the total sorting time (line containing "Sorted" and extract the seconds)
    sort_time=\$(grep "Sorted" temp_output.txt | awk '{print \$(NF-1)}')

    # Save n and sorting time to file
    echo "\$n \$sort_time" >> \$OUTFILE

    # Optionally print to screen
    echo "n=\$n, sort_time=\${sort_time}s"
done

# Clean up temp file
rm temp_output.txt

# Final message
echo "All done. Results saved to \$OUTFILE"
EOF

chmod +x "$SCRIPT"
echo "Generated $SCRIPT"