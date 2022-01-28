#cc_binary(
#    name = "main",
#    srcs = ["main.cc"],
#    deps = ["//:opencv"],
#)

py_library(
    name="filevideostream",
    srcs=["filevideostream.py"],
    # visibility
)

py_binary(
    name = "label_image_py",
    srcs = ["label_image.py"],
    main = "label_image.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps=["//label_image:filevideostream"]
)