---
name: calculator_skill
description: Evaluate arithmetic expressions like (17 * 23) + 9
type: prompt
template: "Evaluate the expression: {expression}\nResult: "
parameters:
  type: object
  properties:
    expression:
      type: string
      description: Math expression to evaluate
  required:
    - expression
---

# Calculator Skill

Evaluates arithmetic expressions.
