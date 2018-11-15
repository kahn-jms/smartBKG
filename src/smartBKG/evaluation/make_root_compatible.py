#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
My version of the code in framework/utilities/MakeROOTCompatible.h
James Kahn
"""

substitution_map = {
    " ": "__sp",
    ",": "__cm",
    ":": "__cl",
    "=": "__eq",
    "<": "__st",
    ">": "__gt",
    ".": "__pt",
    "+": "__pl",
    "-": "__mi",
    "(": "__bo",
    ")": "__bc",
    "{": "__co",
    "}": "__cc",
    "[": "__so",
    "]": "__sc",
    "`": "__to",
    "´": "__tc",
    "^": "__ha",
    "°": "__ci",
    "$": "__do",
    "§": "__pa",
    "%": "__pr",
    "!": "__em",
    "?": "__qm",
    ";": "__sm",
    "#": "__hs",
    "*": "__mu",
    "/": "__sl",
    "\\": "__bl",
    "'": "__sq",
    "\"": "__dq",
    "~": "__ti",
    # "-": "__da",
    "|": "__pi",
    "&": "__am",
    "@": "__at",
}


def makeRootCompatible(root_string):
    for item in substitution_map:
        root_string = root_string.replace(item, substitution_map[item])
    return root_string


def invertMakeRootCompatible(root_string):
    for item in substitution_map:
        root_string = root_string.replace(substitution_map[item], item)
    return root_string


def shorten_longnames(root_string):
    if root_string.startswith('formula'):
        return root_string[8:-1]
    else:
        return root_string
