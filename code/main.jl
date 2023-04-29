#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using LinearAlgebra:diagm
using PyCall
using LaTeXStrings

# %%
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=10)

const γ = 1.4

# %%

function minmod(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
	if sign(a) * sign(b) > 0
		if abs(a) < abs(b)
			return a
		end
		return b
	end
	return 0
end
# %%


function w2U(w::Vector)::Vector
	u=similar(w)
	u[1] = w[1]
	u[2] = w[2] / u[1]
	u[3] = (γ-1) * (w[3] - 0.5 * u[1] * u[2]^2)
	return u
end

function U2L(U::Vector)::Matrix
	ρ = U[1]
	u = U[2]
	p = U[3]
	a = sqrt(γ*p/ρ)
	L = [ 0  -ρ*a 1;
		  a^2 0  -1;
		  0  ρ*a  1]
end

function U2R(U::Vector)::Matrix
	ρ = U[1]
	u = U[2]
	p = U[3]
	a = sqrt(γ*p/ρ)
	R = [ 0.5/a^2  1/a^2 0.5/a^2;
		 -0.5/(ρ*a)  0    0.5/(ρ*a);
		  0.5        0     0.5  ]
end

function w2L(w::Vector)::Matrix
	U = w |> w2U
	ρ = w[1]
	m = w[2]
	u = U[2]
	E = w[3]
	p = U[3]
	a = sqrt(γ*p/ρ)
	H = a^2/(γ-1) + 0.5u^2
	L = 0.5*(γ-1)/a^2 * [ 0.5u*(u+2*a/(γ-1))    -(u+a/(γ-1))    1;
						  2*(H-u^2)                2u          -2;
						  0.5u*(u-2*a/(γ-1))    -(u-a/(γ-1))    1]
end

function w2A(w::Vector)::Matrix
	U = w |> w2U
	ρ = w[1]
	m = w[2]
	u = U[2]
	E = w[3]
	A = [ 0                         1                0 ;
		 0.5u^2*(γ-3)             -u*(γ-3)          γ-1;
		 (γ-1)*u^3-γ*u/ρ*E    γ/ρ*E-1.5*(γ-1)*u^2   γ*u]
end


function w2F(w::Vector)::Vector
	U = w |> w2U
	u = U[2]
	p = U[3]
	F = u*w .+ [0, p, p*u]
end


function w2R(w::Vector)::Matrix
	U = w |> w2U
	ρ = w[1]
	m = w[2]
	u = U[2]
	E = w[3]
	p = U[3]
	a = sqrt(γ*p/ρ)
	H = a^2/(γ-1) + 0.5u^2
	R = [ 1        1        1  ;
		 u-a       u        u+a;
		 H-u*a    0.5u^2   H+u*a]
end

function U2λ(U::Vector)::Vector
	ρ = U[1]
	u = U[2]
	p = U[3]
	a = sqrt(γ*p/ρ)
	λ = [u-a, u, u+a]
end

w2λ(w::Vector)::Vector = w |> w2U |> U2λ

function upwind_non(UP::Matrix, U::Matrix, C::AbstractFloat)
	# l=104
	# sum(U .== NaN)
	# U[:, 104]
	# c.u[:, 104]
	for l in 2:size(U, 2)-1
		λ= U[:, l] |> U2λ
		R = U[:, l] |> U2R
		L = U[:, l] |> U2L
		D=similar(U[:, l])
		for i in 1:3
			D[i] = U[i, l] - U[i, l-Int(sign(λ[i]))]
		end
		Λ= abs.(λ) |> diagm
		UP[:, l] .= U[:, l] - C * R*Λ*L * D
	end
end

function lax_wendroff(wp::Matrix, w::Matrix, C::AbstractFloat)
for l in 2:size(w, 2)-1
	Am = 0.5*(w[:, l]+w[:, l-1]) |> w2A
	Ap = 0.5*(w[:, l]+w[:, l+1]) |> w2A
	Fm = w[:, l-1] |> w2F
	Fp = w[:, l+1] |> w2F
	F = w[:, l] |> w2F
	wp[:, l] .= w[:, l] - 0.5C*(Fp - Fm) +
	0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	# B = 0.5C*(u[l] + u[l-1])
	# up[l] = u[l] - B *  (u[l] - u[l-1]) - 0.5 * B * (1 - B) *
	# 	( minmod(u[l]-u[l-1], u[l+1]-u[l]) - minmod(u[l-1]-u[l-2], u[l]-u[l-1]) )
end
end

function upwind(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		λ= w[:, l] |> w2λ
		R = w[:, l] |> w2R
		L = w[:, l] |> w2L
		D=similar(w[:, l])
		for i in 1:3
			D[i] = w[i, l] - w[i, l-Int(sign(λ[i]))]
		end
		Λ= abs.(λ) |> diagm
		wp[:, l] .= w[:, l] - C * R*Λ*L * D
	end
end

# %%

function init_non(x::AbstractVector, u::Matrix)
	w = [0.445, 0.311, 8.928]
	u[:, x .< 0] .= w2U(w)
	w = [0.5, 0, 1.4275]
	u[:, x .>= 0 ] .= w2U(w)
end

function init0_con(x::AbstractVector, w::Matrix)
	w[:, x .< 0] .= [0.445, 0.311, 8.928]
	w[:, x .>= 0 ] .= [0.5, 0, 1.4275]
end

struct Cells
	x::AbstractVector{Float64}
	u::Matrix{Float64} # u^n
	up::Matrix{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=1.0; step::Float64=0.01, init::Function=init0)
	x = range(b, e, step=step)
	u=zeros(3,length(x))
	init(x, u)
	up=deepcopy(u)
	new(x, u , up)
	end
end

Cells(Δ::Float64)=Cells(-1.0, 1.0, step=Δ)
Cells(init::Function)=Cells(-1.0, 1.0, init=init)
Cells(b::Float64, e::Float64, Δ::Float64)=Cells(b, e, step=Δ)

next(c::Cells, flg::Bool)::Matrix = flg ? c.up : c.u
current(c::Cells, flg::Bool)::Matrix = flg ? c.u : c.up

function update!(c::Cells, flg::Bool, f::Function, C::AbstractFloat)
	UP=next(c, flg) # u^(n+1)
	U=current(c, flg) # u^n
	f(UP, U, C)
	return !flg
end
update!(c::Cells, flg::Bool, f::Function) = update!(c, flg, f, 0.5)

# %%
C = 0.5
Δx= 0.01
# C = Δt/Δx
Δt =  C * Δx



# %%
function problem1(C::AbstractFloat, f::Function, title::String; Δx::AbstractFloat=0.007)

	title = L"$m$"
	# t=0.002
	t=0.14
	C = 0.2
	Δx= 2/261
	Δt = Δx * C
	f = upwind_non
	c=Cells(step=Δx, init=init_non)
	plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update!(c, flg, f, C)
	end
	w=current(c, flg)
	plt.plot(c.x, w[1, :], "-.b", linewidth=1)
	# plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	plt.show()

	plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title )
	# plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	# plt.savefig("../figures/problem1_"*string(f)*string(C)*title*".pdf", bbox_inches="tight")
	plt.show()

end

# %%

function main()
	problem1(0.05, upwind, "Upwind")
	problem1(0.5, upwind, "Upwind")
	problem1(0.95, upwind, "Upwind")
	problem1(1.0, upwind, "Upwind")
	problem1(0.95, lax_wendroff, "Lax-Wendroff")
	problem1(0.95, limiter, "Minmod")
	# problem2(0.25)
	# problem2(0.5)
	# problem2(0.75)
	# problem2(1.0)
end
main()
