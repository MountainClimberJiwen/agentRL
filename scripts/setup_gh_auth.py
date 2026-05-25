#!/usr/bin/env python3
"""
Setup GitHub CLI authentication.

Usage:
    export GH_TOKEN=ghp_your_token_here
    python /opt/agentrl/scripts/setup_gh_auth.py
"""

import os
import subprocess
import sys

def main():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    
    if not token:
        # Try reading from .env
        env_paths = ["/opt/agentrl/.env", os.path.expanduser("~/.hermes/.env"), os.path.expanduser("~/.env")]
        for path in env_paths:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.startswith("GH_TOKEN=") or line.startswith("GITHUB_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if token:
                    break
    
    if not token:
        print("❌ No GH_TOKEN found.")
        print("")
        print("To create a token:")
        print("  1. Visit: https://github.com/settings/tokens/new")
        print("  2. Select scopes: repo, read:org, gist")
        print("  3. Generate and copy the token")
        print("  4. Save it: echo 'GH_TOKEN=ghp_xxx' >> /opt/agentrl/.env")
        print("  5. Run this script again")
        sys.exit(1)
    
    # Configure gh
    config_dir = os.path.expanduser("~/.config/gh")
    os.makedirs(config_dir, exist_ok=True)
    
    with open(os.path.join(config_dir, "hosts.yml"), "w") as f:
        f.write(f"github.com:\n")
        f.write(f"    user: MountainClimberJiwen\n")
        f.write(f"    oauth_token: {token}\n")
        f.write(f"    git_protocol: ssh\n")
    
    with open(os.path.join(config_dir, "config.yml"), "w") as f:
        f.write(f"github.com:\n")
        f.write(f"    user: MountainClimberJiwen\n")
        f.write(f"    oauth_token: {token}\n")
        f.write(f"    git_protocol: ssh\n")
    
    print("✅ gh authentication configured!")
    
    # Verify
    r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=10)
    print("\nAuth status:")
    print(r.stdout if r.stdout else r.stderr)

if __name__ == "__main__":
    main()
