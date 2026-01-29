#!/usr/bin/env python3
"""
Test script to verify dataset configuration and prompt generation.
"""
import json
from prompt_generator import get_dataset_config, get_dataset_tools, generate_system_prompt, load_dataset_config

def test_configuration():
    """Test loading and displaying all dataset configurations."""
    print("="*80)
    print("Testing Dataset Configuration")
    print("="*80)
    
    try:
        config = load_dataset_config()
        datasets = config['datasets']
        
        print(f"\nTotal datasets configured: {len(datasets)}\n")
        
        # Group datasets by tool configuration
        tool_groups = {}
        for dataset_name, dataset_config in datasets.items():
            tools_str = ", ".join(dataset_config['tools'])
            if tools_str not in tool_groups:
                tool_groups[tools_str] = []
            tool_groups[tools_str].append(dataset_name)
        
        print("Datasets grouped by tools:")
        print("-" * 80)
        for tools_str, dataset_list in sorted(tool_groups.items()):
            print(f"\nTools: {tools_str}")
            print(f"Datasets ({len(dataset_list)}):")
            for ds in sorted(dataset_list):
                config = get_dataset_config(ds)
                desc = config.get('description', 'No description')
                max_calls = config.get('max_llm_calls', 'N/A')
                print(f"  - {ds:20s} | Max LLM calls: {max_calls:3} | {desc}")
        
        print("\n" + "="*80)
        print("Configuration test PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nConfiguration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_generation(dataset_name="gaia"):
    """Test prompt generation for a specific dataset."""
    print("\n" + "="*80)
    print(f"Testing Prompt Generation for: {dataset_name}")
    print("="*80)
    
    try:
        tools = get_dataset_tools(dataset_name)
        prompt = generate_system_prompt(tools)
        
        print(f"\nDataset: {dataset_name}")
        print(f"Tools: {tools}")
        print(f"\nGenerated prompt length: {len(prompt)} characters")
        print(f"Number of tool definitions: {len(tools)}")
        
        # Verify all tools are in the prompt
        for tool in tools:
            if f'"name": "{tool}"' in prompt:
                print(f"✓ Tool '{tool}' found in prompt")
            else:
                print(f"✗ Tool '{tool}' NOT found in prompt")
                return False
        
        print("\n" + "="*80)
        print("Prompt generation test PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nPrompt generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_datasets():
    """Test configuration for all datasets."""
    print("\n" + "="*80)
    print("Testing All Dataset Configurations")
    print("="*80)
    
    config = load_dataset_config()
    datasets = list(config['datasets'].keys())
    
    failed = []
    for dataset_name in datasets:
        try:
            dataset_config = get_dataset_config(dataset_name)
            tools = get_dataset_tools(dataset_name)
            prompt = generate_system_prompt(tools)
            
            # Basic validation
            assert len(tools) > 0, "No tools configured"
            assert len(prompt) > 0, "Empty prompt"
            assert 'data_file' in dataset_config, "No data_file specified"
            
            print(f"✓ {dataset_name:20s} - OK")
            
        except Exception as e:
            print(f"✗ {dataset_name:20s} - FAILED: {e}")
            failed.append(dataset_name)
    
    print("\n" + "="*80)
    if failed:
        print(f"Test FAILED for {len(failed)} datasets: {failed}")
        print("="*80)
        return False
    else:
        print(f"All {len(datasets)} datasets tested successfully!")
        print("="*80)
        return True


if __name__ == "__main__":
    import sys
    
    all_passed = True
    
    # Test 1: Configuration loading
    all_passed &= test_configuration()
    
    # Test 2: Prompt generation
    all_passed &= test_prompt_generation("gaia")
    all_passed &= test_prompt_generation("math500")
    all_passed &= test_prompt_generation("2wiki")
    
    # Test 3: All datasets
    all_passed &= test_all_datasets()
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*80)
    
    sys.exit(0 if all_passed else 1)

