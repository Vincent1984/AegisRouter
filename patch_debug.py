"""Patch LiteLLM proxy utils.py to print callbacks at request time."""
src_path = '/usr/local/lib/python3.11/site-packages/litellm/proxy/utils.py'
with open(src_path) as f:
    content = f.read()

old = 'verbose_proxy_logger.debug("Inside Proxy Logging Pre-call hook!")'
new = (
    old + '\n'
    '        import sys as _sys; '
    '_sys.stderr.write(f"[HOOK-DEBUG] callbacks={[type(c).__name__ for c in litellm.callbacks]}\\n"); '
    '_sys.stderr.flush()'
)
content = content.replace(old, new, 1)
with open(src_path, 'w') as f:
    f.write(content)
print('Patched!')
