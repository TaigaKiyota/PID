using Random
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems
include("function.jl")

rng = MersenneTwister(1)
using Zygote
using JLD2