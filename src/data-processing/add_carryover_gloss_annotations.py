import os
import pysrt

# Base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"

def process_folder(folder_path):
    """Process a single folder's subtitle file and create an annotated version."""
    input_path = os.path.join(folder_path, "filtered-transcript.srt")
    output_path = os.path.join(folder_path, "filtered-transcript-carryover-glosses.srt")
    
    # Attempt to open the input SRT file
    try:
        srt_file = pysrt.open(input_path)
    except Exception as e:
        print(f"❌ Error reading transcript file in {folder_path}: {e}")
        return
    
    # Create a new SubRipFile for the output
    output_srt = pysrt.SubRipFile()
    
    # State variables to track carryover conditions
    previous_sentence_speaker = None
    previous_speaker = None
    direct_carryover_mode = False
    
    # Process each subtitle in the input SRT file
    for subtitle in srt_file:
        text = subtitle.text.strip()
        # Validate speaker format (must start with "A:" or "B:")
        if not (text.startswith("A:") or text.startswith("B:")):
            continue
        
        # Split the text into speaker and content
        parts = text.split(":", 1)
        speaker = parts[0].strip()
        content = parts[1].strip()
        
        if content.endswith("_FULL_SENTENCE"):
            # Handle sentences
            if previous_sentence_speaker is not None and speaker != previous_sentence_speaker:
                # Speaker change detected
                previous_speaker = previous_sentence_speaker
                direct_carryover_mode = True
            previous_sentence_speaker = speaker
            # Keep sentence text unchanged
            new_text = text
        else:
            # Handle glosses
            # Check if it's a total carryover gloss
            is_total_carryover = previous_sentence_speaker is not None and speaker != previous_sentence_speaker
            # Check if it's a direct carryover gloss
            is_direct_carryover = is_total_carryover and direct_carryover_mode and speaker == previous_speaker
            
            if is_direct_carryover:
                # Annotate direct carryover gloss
                new_content = content + "_DIRECT_CARRYOVER"
            elif is_total_carryover:
                # Annotate other carryover gloss
                new_content = content + "_CARRYOVER"
            else:
                # No annotation for non-carryover gloss
                new_content = content
            new_text = f"{speaker}: {new_content}"
            
            # If gloss is from the current sentence speaker, stop direct carryover mode
            if speaker == previous_sentence_speaker:
                direct_carryover_mode = False
        
        # Create a new subtitle with the same timing and modified text
        new_subtitle = pysrt.SubRipItem(
            index=subtitle.index,
            start=subtitle.start,
            end=subtitle.end,
            text=new_text
        )
        output_srt.append(new_subtitle)
    
    # Save the annotated SRT file
    try:
        output_srt.save(output_path, encoding='utf-8')
        print(f"🎉 Annotated SRT file created at {output_path}")
    except Exception as e:
        print(f"❌ Error writing annotated SRT file in {folder_path}: {e}")

# Main execution
print("🚀 Starting processing of DGSKorpus folders...")
folders = [entry for entry in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, entry))]
total_folders = len(folders)
processed_folders = 0

for entry in folders:
    folder_path = os.path.join(base_path, entry)
    processed_folders += 1
    progress = (processed_folders / total_folders) * 100
    print(f"📂 Processing folder: {folder_path} ({processed_folders}/{total_folders}, {progress:.1f}% complete)")
    process_folder(folder_path)

print("✅ Processing complete!")