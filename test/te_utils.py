from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

class ExecutionCounter(object):
    def try_get_trigger_value(self):
        try:
            return torch._C._jit_get_trigger_value(self.name)
        except Exception:
            return 0

    def __init__(self, name):
        self.name = name
        self.start_value = self.try_get_trigger_value()

    def elapsed_value(self):
        value = self.try_get_trigger_value()
        return value - self.start_value

class CudaCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super().__init__("cuda_codegen_created")

class CudaCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super().__init__("cuda_codegen_executed")

class LLVMCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super().__init__("llvm_codegen_created")

class LLVMCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super().__init__("llvm_codegen_executed")

class SimpleIREvalExecuted(ExecutionCounter):
    def __init__(self):
        super().__init__("simple_ir_eval_executed")
