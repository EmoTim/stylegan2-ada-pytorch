import boto3
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from loguru import logger

bucket = "emobot-prod-workspace-bucket"
prefix = "usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/"

console = Console()
s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")

# Dictionary to store counts per alpha folder
alpha_counts = defaultdict(int)
total_count = 0
pages = 0

logger.info("Starting S3 object enumeration for bucket: {}, prefix: {}", bucket, prefix)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Scanning S3 objects...", total=None)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])

        for obj in contents:
            key = obj["Key"]
            # Extract alpha folder from path like: .../images/alpha_-2.0/seed0001.jpg
            parts = key.replace(prefix, "").split("/")
            if len(parts) >= 2 and parts[0].startswith("alpha_"):
                alpha_folder = parts[0]
                alpha_counts[alpha_folder] += 1
                total_count += 1

        pages += 1
        progress.update(task, description=f"[cyan]Scanning S3 objects... ({pages} pages, {total_count} images)")

logger.success("Scan complete! Processed {} pages, found {} total images", pages, total_count)

# Create a rich table to display results
table = Table(title="Image Count by Alpha Folder", show_header=True, header_style="bold magenta")
table.add_column("Alpha Folder", style="cyan", no_wrap=True)
table.add_column("Image Count", justify="right", style="green")

# Sort alpha folders by their numeric value
sorted_alphas = sorted(alpha_counts.items(), key=lambda x: float(x[0].replace("alpha_", "")))

for alpha_folder, count in sorted_alphas:
    table.add_row(alpha_folder, str(count))

# Add total row
table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_count}[/bold]", style="bold yellow")

console.print("\n")
console.print(table)
console.print(f"\n[bold green]Total images across all alpha folders:[/bold green] {total_count}")
