# CVL Guidelines

This file used to be a standalone copy of the 23 CVL authoring rules AI Composer's CVL Judge enforces. It was
identical to the source at the time (differing only in an XML wrapper), but a hand-maintained duplicate has no
way to stay in sync if the real one changes — so it's been replaced with a pointer instead of a second copy.

**The authoritative source is [`composer/templates/cvl_guidelines.j2`](composer/templates/cvl_guidelines.j2).**
Read that file directly for the current, enforced list of rules.

If you need this content in prose form for a document that can't include Jinja templates, generate it fresh from
the template rather than hand-copying it, e.g.:

```bash
python3 -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('composer/templates'))
print(env.get_template('cvl_guidelines.j2').render())
"
```
