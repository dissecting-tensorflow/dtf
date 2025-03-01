# String with octal escapes
octal_str = r"\001\000\000\000\002\000\000\000\003\000\000\000"
items = octal_str.split("\\")[1:]
for it in items:
  print(int(it, 8))

# Convert to bytes (implicitly interprets the escapes)
# byte_form = octal_str.encode()
# print(byte_form)