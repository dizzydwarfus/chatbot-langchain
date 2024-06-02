import re
import streamlit as st

st.title("Logs")

with st.expander("ℹ About"):
    st.info(
        "This page displays the logs from the chatbot. The most recent logs are shown first."
    )

with st.sidebar:
    st.info("Upload a log file to view its contents.")
    uploaded_file = st.file_uploader("Upload log file", type=["log"])

if uploaded_file:
    log_lines = uploaded_file.getvalue().decode("utf-8").splitlines()[::-1]

    log_groups = []
    current_group = []

    for line in log_lines:
        # Check if the line starts with a timestamp
        if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", line):
            # If it does, this is the start of a new log statement
            # Add the current group to the list of groups and start a new group
            current_group = [line]
            log_groups.append(current_group)
        else:
            # If it doesn't, this line is a continuation of the current log statement
            current_group.append(line)

    # Add the last group to the list of groups
    # if current_group:
    #     log_groups.append(current_group)
    # Display each group of log lines as a separate block of preformatted text
    for group in log_groups:
        group_text = "\n".join(group)
        if re.search(
            r".*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}.*- INFO -.*", group_text
        ):
            st.info(f"```\n{group_text}\n```")
        elif re.search(
            r".*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}.*- ERROR -.*", group_text
        ):
            st.error(f"```\n{group_text}\n```")
        else:
            st.warning(f"```\n{group_text}\n```")
