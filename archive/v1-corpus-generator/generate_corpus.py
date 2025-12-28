#!/usr/bin/env python3
"""Main script to generate the CPRA email corpus with ground truth."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.generators.school_district import SchoolDistrictGenerator
from src.generators.cpra_requests import CPRARequestGenerator
from src.generators.email_generator import EmailGenerator, EmailGenerationConfig
from src.utils.data_export import DataExporter
from src.utils.llm_client import LLMClient, LLMConfig

# Load environment variables from .env file
load_dotenv()


def load_yaml_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "generation_config.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def build_config_from_yaml(yaml_config: dict, args) -> EmailGenerationConfig:
    """Build EmailGenerationConfig from YAML config and CLI args."""
    gen = yaml_config.get('generation', {})
    email = yaml_config.get('email', {})
    llm = yaml_config.get('llm', {})

    return EmailGenerationConfig(
        total_emails=args.num_emails if args.num_emails != 2500 else gen.get('total_emails', 2500),
        responsive_rate=args.responsive_rate if args.responsive_rate != 0.15 else gen.get('responsive_rate', 0.15),
        challenge_email_rate=args.challenge_rate if args.challenge_rate != 0.3 else gen.get('challenge_email_rate', 0.3),
        min_email_length=email.get('min_length', 50),
        max_email_length=email.get('max_length', 500),
        attachment_probability=email.get('attachment_probability', 0.1),
        thread_probability=email.get('thread_probability', 0.25),
        cc_probability=email.get('cc_probability', 0.3),
        use_llm=args.use_llm or llm.get('use_llm', False),
        llm_provider=args.llm_provider or llm.get('provider', 'openai'),
        llm_model=llm.get('model', 'gpt-4'),
        llm_temperature=llm.get('temperature', 0.7),
        llm_max_tokens=llm.get('max_tokens', 500),
    )


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate CPRA email corpus with ground truth")
    parser.add_argument(
        "--num-emails",
        type=int,
        default=2500,
        help="Number of emails to generate (default: 2500)"
    )
    parser.add_argument(
        "--responsive-rate",
        type=float,
        default=0.15,
        help="Percentage of emails that should be responsive (default: 0.15)"
    )
    parser.add_argument(
        "--challenge-rate",
        type=float,
        default=0.3,
        help="Percentage of responsive emails with challenge patterns (default: 0.3)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for email generation (requires API keys)"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider to use (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated",
        help="Output directory for generated corpus (default: data/generated)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )

    args = parser.parse_args()

    # Load YAML configuration
    yaml_config = load_yaml_config(args.config)
    config = build_config_from_yaml(yaml_config, args)

    print("=" * 60)
    print("CPRA Email Corpus Generator")
    print("=" * 60)
    print(f"Generating {config.total_emails} emails with {config.responsive_rate:.1%} responsive rate")
    print(f"Challenge patterns in {config.challenge_email_rate:.1%} of responsive emails")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    if config.use_llm:
        print(f"LLM: {config.llm_provider} / {config.llm_model}")
    else:
        print("LLM: disabled (using templates)")
    print()

    # Step 1: Generate school district context
    print("Step 1: Generating school district context...")
    district_generator = SchoolDistrictGenerator(seed=args.seed)
    district = district_generator.generate_district()
    print(f"  ✓ Created {len(district.schools)} schools")
    print(f"  ✓ Generated {len(district.staff)} staff members")
    print(f"  ✓ Established {len(district.departments)} departments")
    print()

    # Step 2: Generate CPRA requests
    print("Step 2: Creating CPRA requests...")
    request_generator = CPRARequestGenerator()
    request_set = request_generator.generate_requests()
    print(f"  ✓ Generated {len(request_set.requests)} CPRA requests")
    for request in request_set.requests:
        print(f"    - {request.title} ({request.complexity.value})")
    print()

    # Step 3: Generate email corpus
    print("Step 3: Generating email corpus...")
    email_generator = EmailGenerator(
        district=district,
        requests=request_set.requests,
        config=config
    )

    # Add progress bar for email generation
    print("  Generating emails...")
    ground_truth = email_generator.generate_corpus()

    print(f"  ✓ Generated {ground_truth.total_emails} emails")
    print(f"  ✓ {ground_truth.responsive_emails} responsive emails ({ground_truth.responsive_emails/ground_truth.total_emails:.1%})")

    # Show responsiveness breakdown
    stats = ground_truth.get_statistics()
    print("\n  Responsiveness by request:")
    for request in request_set.requests:
        count = stats["emails_per_request"].get(request.id, 0)
        print(f"    - {request.title}: {count} emails")

    print("\n  Challenge pattern distribution:")
    for pattern, count in stats.get("challenge_distribution", {}).items():
        if count > 0:
            print(f"    - {pattern}: {count} emails")
    print()

    # Step 4: Export data
    print("Step 4: Exporting data...")
    exporter = DataExporter(output_dir=args.output_dir)
    exported_files = exporter.export_all(
        ground_truth=ground_truth,
        district=district,
        requests=request_set.requests
    )

    print("  ✓ Exported files:")
    for file_type, path in exported_files.items():
        print(f"    - {file_type}: {path}")
    print()

    # Step 5: Generate summary report
    print("Step 5: Generating summary report...")
    summary = {
        "generation_date": datetime.now().isoformat(),
        "configuration": {
            "total_emails": config.total_emails,
            "responsive_rate": config.responsive_rate,
            "challenge_email_rate": config.challenge_email_rate,
            "use_llm": config.use_llm,
            "seed": args.seed
        },
        "results": {
            "total_emails": ground_truth.total_emails,
            "responsive_emails": ground_truth.responsive_emails,
            "response_rate": ground_truth.responsive_emails / ground_truth.total_emails if ground_truth.total_emails > 0 else 0,
            "cpra_requests": len(request_set.requests),
            "staff_members": len(district.staff),
            "schools": len(district.schools)
        },
        "output_location": str(exported_files.get("ground_truth_json", "").parent)
    }

    summary_path = Path(exported_files.get("ground_truth_json", "")).parent / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Summary saved to {summary_path}")
    print()

    print("=" * 60)
    print("✨ Generation complete!")
    print(f"📁 Output directory: {summary_path.parent}")
    print("=" * 60)

    # Print next steps
    print("\n📝 Next Steps:")
    print("1. Review generated emails in:", exported_files.get("emails_dir", ""))
    print("2. Examine ground truth in:", exported_files.get("ground_truth_json", ""))
    print("3. Analyze data in Excel:", exported_files.get("excel", ""))
    print("\nYou can now use this corpus to test different CPRA responsiveness detection techniques!")


if __name__ == "__main__":
    main()