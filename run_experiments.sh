#!/bin/bash

# Function to run a command in a new tmux session
run_in_tmux() {
    session_name="exp_$1"
    tmux new-session -d -s "$session_name" bash -c "
        source ~/.bashrc;
        conda activate FL_drone;
        $2;
        exec bash"
}

# Define experiments
declare -a experiments=(
    "export CLUSTERING='energy'; export K1_ALLOCATION='reversed'; python main.py"
    "export CLUSTERING='energy'; export K1_ALLOCATION='non_reversed'; python main.py"
    "export CLUSTERING='energy'; export K1_ALLOCATION='uniform'; python main.py"
)

for i in "${!experiments[@]}"; do
    echo "Starting Experiment $((i+1)) in a new tmux session..."
    run_in_tmux "$i" "${experiments[i]}"
    
    # Wait 5 minutes before starting the next one
    if [[ $i -lt $((${#experiments[@]} - 1)) ]]; then
        echo "Waiting 1 minutes before starting the next experiment..."
        sleep 60
    fi
done

echo "All experiments started in tmux sessions!"

# Show all running tmux sessions
tmux list-sessions
