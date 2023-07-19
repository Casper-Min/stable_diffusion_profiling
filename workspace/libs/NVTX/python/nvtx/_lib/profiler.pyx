import os
import  sys
import threading

from nvtx._lib import (
    push_range as libnvtx_push_range,
    pop_range as libnvtx_pop_range
)

from nvtx._lib.lib cimport EventAttributes, DomainHandle


cdef class Profile:

    def __init__(self, linenos=True, annotate_cfuncs=True):
        """
        Class that enables turning on and off automatic annotation.

        Parameters
        ----------
        linenos: bool (default True)
            Include file and line number information in annotations.
        annotate_cfuncs: bool (default False)
            Also annotate C-extensions and builtin functions.

        Function Name or File Name Exception List
        _contextlib.py:112(decorate_context)
        pipeline_stable_diffusion_profile.py:544(__call__)
        module.py:1494(_call_impl)
        nvtx.py:136(push_range)
        cached.py:10(__call__)
        isinstance
        push_range
        _get_tracing_state
        module.py:1601(__getattr__)
        container.py:311(__len__)
        functional.py:2518(group_norm)
        linear.py:113(forward)
        attention_processor.py:873(__call__)
        conv.py:454(_conv_forward)
        functional.py:2518(group_norm)
        Examples
        --------
        >>> import nvtx
        >>> import time
        >>> pr = nvtx.Profile()
        >>> pr.enable()
        >>> time.sleep(1) # this call to `sleep` is captured by nvtx.
        >>> pr.disable()
        >>> time.sleep(1) # this one is not.
        """
        self.linenos = linenos
        self.annotate_cfuncs = annotate_cfuncs
        self.__domain = DomainHandle("Profiler_AUTO")
        self.__attrib = EventAttributes("", "green", None)

    def _profile(self, frame, event, arg):

        if ( os.path.basename(frame.f_code.co_filename) not in \
            ["module.py","nvtx.py","_contextlib.py","cached.py","grad_mode.py","threading.py","_monitor.py"] ) \
            and ( frame.f_code.co_name not in ["decorate_context","_call_impl", "_call_"] ) :
            # profile function meant to be used with sys.setprofile
            if event == "call":
                name = frame.f_code.co_name
                if self.linenos:
                    fname = os.path.basename(frame.f_code.co_filename)
                    lineno = frame.f_lineno
                    message = f"{fname}:{lineno}({name})"
                else:
                    message = name
                self.push_range(message)
            elif event == "c_call" and self.annotate_cfuncs:
                self.push_range(arg.__name__)
            elif event == "return":
                self.pop_range()
            elif event in {"c_return", "c_exception"} and self.annotate_cfuncs:
                self.pop_range()
        return None

    cdef push_range(self, message):
        self.__attrib.message = message
        libnvtx_push_range(self.__attrib, self.__domain)

    cdef pop_range(self):
        libnvtx_pop_range(self.__domain)

    def enable(self):
        """Start annotating function calls automatically.
        """
        threading.setprofile(self._profile)
        sys.setprofile(self._profile)

    def disable(self):
        """Stop annotating function calls automatically.
        """
        threading.setprofile(None)
        sys.setprofile(None)
