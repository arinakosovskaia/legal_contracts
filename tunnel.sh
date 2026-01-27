#!/bin/bash
# Quick script to start Cloudflare tunnel for the contract detector
# Usage: ./tunnel.sh

echo "Starting Cloudflare tunnel for http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

cloudflared tunnel --url http://localhost:8000
