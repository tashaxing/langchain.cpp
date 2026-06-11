---
name: greeter
description: Generate a greeting for a given name and language
type: prompt
template: "Write a greeting for {name} in {language}."
parameters:
  type: object
  properties:
    name:
      type: string
      description: Name of the person to greet
    language:
      type: string
      description: Language for the greeting
  required:
    - name
    - language
---

# Greeter Skill

Generates a greeting message.
