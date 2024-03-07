filename="/kepler/lightPred/kepler_lightcurves_Q04_long.sh"  # Replace with your file name
start_line=42356             # Replace with the line number from which you want to start executing

tail -n +$start_line "$filename" | while IFS= read -r line; do
    eval "$line"
done