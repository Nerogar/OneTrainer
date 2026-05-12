--- a/util.py
+++ b/util.py
@@ -1,5 +1,5 @@
-def validate_directory_name(name):
-    reserved = ['debug', 'cache']
+def validate_directory_name(name):
+    reserved = ['debug']
     if name in reserved:
         raise ValueError(f"'{name}' is a reserved directory name")
     return name
