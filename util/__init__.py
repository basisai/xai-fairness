import base64
import os
import re
from pathlib import Path

import streamlit as st


def debug_mode():
    st.markdown("<style>* { outline: 1px solid cyan; }</style>", unsafe_allow_html=True)


def uri_encode_path(path, mime="image/png"):
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def interpolate(src, variables={}, rel="."):
    ret = src

    b64_matches = re.findall(r"{{b64:png:(.+?)}}", src)
    for path in b64_matches:
        relpath = os.path.join(rel, path)
        ret = ret.replace(f"{{{{b64:png:{path}}}}}", f"{uri_encode_path(relpath)}")

    text_matches = re.findall(r"{{text:(.+?)}}", src)
    for varname in text_matches:
        ret = ret.replace(f"{{{{text:{varname}}}}}", variables.get(varname) or varname)

    return ret


def include_style(pathstr):
    p = Path(pathstr)
    relpath = os.path.relpath(p.parent)
    interpolated = interpolate(p.read_text(), rel=relpath)
    st.markdown(f"<style>{interpolated}</style>", unsafe_allow_html=True)


def include_html(pathstr, variables={}):
    p = Path(pathstr)
    html = p.read_text()
    relpath = os.path.relpath(p.parent)
    st.markdown(
        interpolate(html, variables=variables, rel=relpath), unsafe_allow_html=True
    )


def page_break():
    st.markdown(
        "<div style='page-break-after: always; line-height: 0;'>&nbsp;</div><div style='height: 0.0001px'>&nbsp;</div>",
        unsafe_allow_html=True,
    )


def add_header(path):
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )
