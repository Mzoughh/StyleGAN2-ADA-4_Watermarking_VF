#!/bin/bash

# =================================================================
# Master script to run all watermarking methods sequentially
# =================================================================

echo "=========================================="
echo "Starting all watermarking methods"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the scripts to run in order
SCRIPTS=(
    "ft_ipr_method.sh"
    "ft_T4G_method.sh"
    "ft_tondi_method.sh"
    "ft_UCHI_method.sh"
)

# Initialize counters
TOTAL=${#SCRIPTS[@]}
CURRENT=0
FAILED=0

# Function to run a script and check its status
run_script() {
    local script_name=$1
    local script_path="$SCRIPT_DIR/$script_name"
    
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Running: $script_name"
    echo "=========================================="
    echo ""
    
    # Make sure the script is executable
    chmod +x "$script_path"
    
    # Run the script
    bash "$script_path"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $script_name completed successfully"
        echo ""
    else
        echo ""
        echo "✗ $script_name failed with error code $?"
        echo ""
        FAILED=$((FAILED + 1))
        
        # Ask if we should continue
        read -p "Do you want to continue with the next script? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping execution..."
            exit 1
        fi
    fi
}

# Start time
START_TIME=$(date +%s)

# Run all scripts sequentially
for script in "${SCRIPTS[@]}"; do
    run_script "$script"
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Final summary
echo ""
echo "=========================================="
echo "           EXECUTION SUMMARY"
echo "=========================================="
echo "Total scripts run: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "Total duration: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "=========================================="
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All methods completed successfully!"
    exit 0
else
    echo "⚠ Some methods failed. Please check the logs."
    exit 1
fi
