add_library(kleidiai STATIC kernel.S ordinary.c generic.S)
set_source_files_properties(kernel.S PROPERTIES LANGUAGE ASM_MARMASM)
set_source_files_properties(ordinary.c PROPERTIES LANGUAGE C)
set_source_files_properties(generic.S PROPERTIES LANGUAGE ASM)
set_source_files_properties(kernel.S ordinary.c generic.S
    PROPERTIES VS_SETTINGS "ExistingMetadata=retained")
