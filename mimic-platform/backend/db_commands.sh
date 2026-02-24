#!/bin/bash
# Database management commands for MIMIC Platform

set -e

echo "🗄️  MIMIC Platform Database Commands"
echo "===================================="
echo ""

case "$1" in
    fix)
        echo "🔧 Fixing database schema (non-destructive)..."
        python fix_database.py
        ;;
    reset)
        echo "🔄 Resetting database (destructive)..."
        python reset_database.py
        ;;
    migrate)
        echo "📦 Running Alembic migrations..."
        alembic upgrade head
        ;;
    create-migration)
        if [ -z "$2" ]; then
            echo "❌ Error: Migration message required"
            echo "Usage: ./db_commands.sh create-migration 'your message'"
            exit 1
        fi
        echo "📝 Creating new migration: $2"
        alembic revision --autogenerate -m "$2"
        ;;
    history)
        echo "📜 Migration history:"
        alembic history
        ;;
    current)
        echo "📍 Current migration:"
        alembic current
        ;;
    downgrade)
        echo "⬇️  Downgrading one version..."
        alembic downgrade -1
        ;;
    psql)
        echo "🐘 Connecting to PostgreSQL..."
        psql -U postgres -d mimic
        ;;
    create-db)
        echo "🏗️  Creating database..."
        psql -U postgres -c "CREATE DATABASE mimic;"
        echo "✅ Database 'mimic' created"
        ;;
    drop-db)
        echo "⚠️  WARNING: This will delete the entire database!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            psql -U postgres -c "DROP DATABASE IF EXISTS mimic;"
            echo "✅ Database 'mimic' dropped"
        else
            echo "❌ Operation cancelled"
        fi
        ;;
    *)
        echo "Available commands:"
        echo ""
        echo "  fix              - Fix schema issues (adds missing columns)"
        echo "  reset            - Reset database (drops and recreates all tables)"
        echo "  migrate          - Run pending Alembic migrations"
        echo "  create-migration - Create new migration (requires message)"
        echo "  history          - Show migration history"
        echo "  current          - Show current migration version"
        echo "  downgrade        - Downgrade one migration version"
        echo "  psql             - Connect to database with psql"
        echo "  create-db        - Create the mimic database"
        echo "  drop-db          - Drop the mimic database"
        echo ""
        echo "Examples:"
        echo "  ./db_commands.sh fix"
        echo "  ./db_commands.sh create-migration 'Add user preferences'"
        echo "  ./db_commands.sh migrate"
        echo ""
        exit 1
        ;;
esac
