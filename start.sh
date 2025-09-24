#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=production
gunicorn app:app --bind 0.0.0.0:10000
