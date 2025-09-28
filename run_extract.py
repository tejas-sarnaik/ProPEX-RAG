from facts_triplet_entity_processor import TripletEntityProcessor

def run_entity_and_triple_extraction() -> bool:
    try:
        processor = TripletEntityProcessor(
            input_file_path="./datasets/sample_database_corpus.json",
            output_dir="./output_directory/output_entity_facts_triplets",
            batch_size=50
        )
        processor.run_complete_processing()
        return True
    except Exception as e:
        print(f"❌ Error in entity/triple extraction: {e}")
        return False

if __name__ == "__main__":
    run_entity_and_triple_extraction()
