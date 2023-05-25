#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using PyCall
using LaTeXStrings

# %%
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=14)

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

function Minmod(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
	a <= 0.0 ? 0.0 : sign(a)*min(abs(a), abs(b))
end


function Minmod(a::AbstractVector, b::AbstractVector)::AbstractVector
	rst = similar(a);
	for i in eachindex(a)
		rst[i] = a[i] <= 0.0 ? 0.0 : sign(a[i])*min(abs(a[i]), abs(b[i]))
	end
	return rst
end

function w2U(w::Vector)::Vector
	u=similar(w)
	u[1] = w[1]
	u[2] = w[2] / u[1]
	u[3] = (γ-1) * (w[3] - 0.5 * u[1] * u[2]^2)
	return u
end
function w2W(w::Matrix)::Matrix
	U = similar(w)
	for l = 1:size(U, 2)
		U[:, l] .= w[:, l] |> w2U
	end
	return U
end

function U2w(U::Vector)::Vector
	w=similar(U)
	ρ = U[1]
	u = U[2]
	p = U[3]
	w[1] = ρ
	w[2] = ρ*u
	w[3] = p/(γ-1) + 0.5ρ*u^2
	return w
end
function U2w(U::Matrix)::Matrix
	w = similar(U)
	for l = 1:size(U, 2)
		w[:, l] .= U[:, l] |> U2w
	end
	return w
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

function lax_wendroff(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		Am = 0.5*(w[:, l]+w[:, l-1]) |> w2A
		Ap = 0.5*(w[:, l]+w[:, l+1]) |> w2A
		Fm = w[:, l-1] |> w2F
		Fp = w[:, l+1] |> w2F
		F = w[:, l] |> w2F
		wp[:, l] .= w[:, l] - 0.5C*(Fp - Fm) +
		0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	end
end

function ygs(up::CircularVector, u::CircularVector, C::AbstractFloat)
	n = length(u)
	cc = zeros(n)
	for i in eachindex(cc)
		cc[i] = 0.5*C*(u[i] + u[i-1])
	end
	A=Matrix(Tridiagonal(-cc[2:end], cc.+1, repeat([0.0], n-1)))
	A[1, n] = -cc[1]
	up .=  A \ u
end

function ygs(wp::Matrix, w::Matrix, C::AbstractFloat)
	v_left = ones(3)
	v_mid
	v_right
	Tridiagonal()
	for l in 2:size(w, 2)-1
		Am = 0.5*(w[:, l]+w[:, l-1]) |> w2A
		Ap = 0.5*(w[:, l]+w[:, l+1]) |> w2A
		Fm = w[:, l-1] |> w2F
		Fp = w[:, l+1] |> w2F
		F = w[:, l] |> w2F
		wp[:, l] .= w[:, l] - 0.5C*(Fp - Fm) +
		0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	end
end

# λ=[1,2,3.]
# w = [1 2 3; 2 2 4]'
# Δw = [0, 3, 5.]

# l = 1

Q(x) = abs(x) > 0.2 ? abs(x) : x^2/0.4 + 0.1

function testfor()
	last_i = 0
	for i = 1:3
		print(last_i-i)
		last_i = i
	end
end

function testfor()
	last_a = 0
	global b = 0
	for i = 1:3
		a = i
		global b = last_a - a
		last_a = a
	end
	@show b
end





function TVD(wp::Matrix, w::Matrix, C::AbstractFloat)
	gt_ = [0., 0, 0]# last g_tilde
	for l in 2:size(w, 2)-1
		Lm = w[:, l] |> w2L
		Lp = w[:, l+1] |> w2L
		L = 0.5*(Lm + Lp) # L_{j+1/2}
		Δw = (w[:, l+1] - w[:, l])
		α = L * Δw # α 是 3x1 向量
		λ = w[:, l] |> w2λ
		ν = simliar(λ)
		g = simliar(λ)
		gt = simliar(λ) # g_tilde
		a = (w2F(w[:, l+1]) - w2F(w[:, l]))
		for k in 1:3
			Δw[k] == 0 && a[k] = λ[k] || a[k] /= Δw[k] # a 是 3x1 向量
			ν = C*a[k]
			gt[k] = 0.5*(Q(ν)-ν^2)*α[k] # g_tilde
			s = sign(gt[k])
			g[k] = s * max(0, min(abs(gt[k]), gt_[k] * s))
			gt_[k] = gt[k]
		end


		Ap = 0.5*(w[:, l]+w[:, l+1]) |> w2A
		Fm = w[:, l-1] |> w2F
		Fp = w[:, l+1] |> w2F
		F = w[:, l] |> w2F
		wp[:, l] .= w[:, l] - 0.5C*(Fp - Fm) +
		0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	end
end

function limiter(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 3:size(w, 2)-2
		λ= w[:, l] |> w2λ
		# R = w[:, l] |> w2R
		# L = w[:, l] |> w2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Λ = 0.5*(λ+λp)
				Rm = w[:, l] |> w2R
				Rp = w[:, l-s] |> w2R
				R = 0.5*(Rm+Rp)
				Lm = w[:, l] |> w2L
				Lp = w[:, l-s] |> w2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					a = Λ[i]*R[k, i]*L[i, j]
					Σ += s*a*(w[j, l] - w[j, l-s])
					Σ += 0.5s*a* (1 - a*C) *
					( minmod(w[j, l]-w[j, l-1], w[j, l+1]-w[j,l]) -
					 minmod(w[j,l-s]-w[j,l-s-1], w[j,l-s+1]-w[j,l-s]) )
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function NND(wp::Matrix, w::Matrix, C::AbstractFloat)
for l in 3:size(w, 2)-2
	F = w[:, l] |> w2F
	Fp1 = w[:, l+1] |> w2F
	Fm1 = w[:, l-1] |> w2F
	Fp2 = w[:, l+2] |> w2F
	Fpp = F + 0.5Minmod(F - Fm1, Fp1 - F)
	Fpm = F - 0.5Minmod(Fp1 - F, Fp2 - Fp1)
	Fp = 0.5(Fpp + Fpm)
	F = w[:, l-1] |> w2F
	Fp1 = w[:, l-1+1] |> w2F
	Fm1 = w[:, l-1-1] |> w2F
	Fp2 = w[:, l-1+2] |> w2F
	Fmp = F + 0.5Minmod(F - Fm1, Fp1 - F)
	Fmm = F - 0.5Minmod(Fp1 - F, Fp2 - Fp1)
	Fm = 0.5(Fmp + Fmm)
	wp[:, l] = w[:, l] - C * (Fp - Fm)
end
end

function upwind_non(UP::Matrix, U::Matrix, C::AbstractFloat)
	for l in 2:size(U, 2)-1
		λ= U[:, l] |> U2λ
		Rm = U[:, l] |> U2R
		Lm = U[:, l] |> U2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = U[:, l-s] |> U2λ
				Λ = 0.5*(λ+λp)
				Rp = U[:, l-s] |> U2R
				R = 0.5*(Rm+Rp)
				Lp = U[:, l-s] |> U2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					a = s*Λ[i]*R[k, i]*L[i, j]
					Σ += a*(U[j, l] - U[j, l-s])
				end
			end
			UP[k, l] =U[k, l] - C*Σ
		end
	end
end

function upwind_non0(UP::Matrix, U::Matrix, C::AbstractFloat)
	for l in 2:size(U, 2)-1
		λ= U[:, l] |> U2λ
		Rm = U[:, l] |> U2R
		Lm = U[:, l] |> U2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = U[:, l-s] |> U2λ
				Rp = U[:, l-s] |> U2R
				# R = 0.5*(Rm+Rp)
				Lp = U[:, l-s] |> U2L
				# L = 0.5*(Lm+Lp)
				for j = 1:3
					am = s*λ[i]*Rm[k, i]*Lm[i, j]
					ap = s*λp[i]*Rp[k, i]*Lp[i, j]
					a = 0.5*(am+ap)
					Σ += a*(U[j, l] - U[j, l-s])
				end
			end
			UP[k, l] =U[k, l] - C*Σ
		end
	end
end

function upwind(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		λ = w[:, l] |> w2λ
		Rm = w[:, l] |> w2R
		Lm = w[:, l] |> w2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Λ = 0.5*(λ+λp)
				Rp = w[:, l-s] |> w2R
				R = 0.5*(Rm+Rp)
				Lp = w[:, l-s] |> w2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					a = s*Λ[i]*R[k, i]*L[i, j]
					Σ += a*(w[j, l] - w[j, l-s])
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function upwind01(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		# λ = w[:, l-1] |> w2λ
		# λp = w[:, l+1] |> w2λ
		# λ = 0.5*(λm+λp)
		# Rm = w[:, l-1] |> w2R
		# Rp = w[:, l+1] |> w2R
		# R = 0.5*(Rm+Rp)
		# Lm = w[:, l-1] |> w2L
		# Lp = w[:, l+1] |> w2L
		# L = 0.5*(Lm+Lp)
		λ = w[:, l] |> w2λ
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Λ = 0.5(λ+λp)
				Rm = w[:, l] |> w2R
				Rp = w[:, l-s] |> w2R
				R = 0.5*(Rm+Rp)
				Lm = w[:, l] |> w2L
				Lp = w[:, l-s] |> w2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					Σ += s*λ[i]*R[k, i]*L[i, j]*(w[j, l] - w[j, l-s])
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function upwind00(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		λ = w[:, l] |> w2λ
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Λ = 0.5(λ+λp)
				Rm = w[:, l] |> w2R
				Rp = w[:, l-s] |> w2R
				R = 0.5*(Rm+Rp)
				Lm = w[:, l] |> w2L
				Lp = w[:, l-s] |> w2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					a = s*Λ[i]*Rm[k, i]*Lm[i, j]
					Σ += a*(w[j, l] - w[j, l-s])
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function upwind0(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		λ = w[:, l] |> w2λ
		Rm = w[:, l] |> w2R
		Lm = w[:, l] |> w2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Rp = w[:, l-s] |> w2R
				# R = 0.5*(Rm+Rp)
				Lp = w[:, l-s] |> w2L
				# L = 0.5*(Lm+Lp)
				for j = 1:3
					am = s*λ[i]*Rm[k, i]*Lm[i, j]
					ap = s*λp[i]*Rp[k, i]*Lp[i, j]
					a = 0.5*(am+ap)
					Σ += a*(w[j, l] - w[j, l-s])
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function limiter0(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 3:size(w, 2)-2
		λ = w[:, l] |> w2λ
		Rm = w[:, l] |> w2R
		Lm = w[:, l] |> w2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Rp = w[:, l-s] |> w2R
				# R = 0.5*(Rm+Rp)
				Lp = w[:, l-s] |> w2L
				# L = 0.5*(Lm+Lp)
				for j = 1:3
					am = s*λ[i]*Rm[k, i]*Lm[i, j]
					ap = s*λp[i]*Rp[k, i]*Lp[i, j]
					if s > 0
						a = 0.6am+0.4ap
					else
						a = 0.5*(am+ap)
					end
					Σ += a*(w[j, l] - w[j, l-s])
					Σ += 0.5a* (1 - a*C) *
					( minmod(w[j, l]-w[j, l-1], w[j, l+1]-w[j,l]) -
					 minmod(w[j,l-s]-w[j,l-s-1], w[j,l-s+1]-w[j,l-s]) )
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end


function limiter00(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 3:size(w, 2)-2
		λ = w[:, l] |> w2λ
		λm = w[:, l-1] |> w2λ
		λp = w[:, l+1] |> w2λ
		Λ = 0.5*(λm+λp)
		Rm = w[:, l-1] |> w2R
		Rp = w[:, l+1] |> w2R
		R = 0.5*(Rm+Rp)
		Lm = w[:, l-1] |> w2L
		Lp = w[:, l+1] |> w2L
		L = 0.5*(Lm+Lp)
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				for j = 1:3
					a = s*Λ[i]*Rm[k, i]*Lm[i, j]
					Σ += a*(w[j, l] - w[j, l-s])
					Σ += 0.5a* (1 - a*C) *
					( minmod(w[j, l]-w[j, l-1], w[j, l+1]-w[j,l]) -
					 minmod(w[j,l-s]-w[j,l-s-1], w[j,l-s+1]-w[j,l-s]) )
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

function limiter01(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 3:size(w, 2)-2
		λ= w[:, l] |> w2λ
		# R = w[:, l] |> w2R
		# L = w[:, l] |> w2L
		for k = 1:3
			Σ=0.0
			for i = 1:3
				s = λ[i] >= 0 ? 1 : -1
				λp = w[:, l-s] |> w2λ
				Λ = 0.5*(λ+λp)
				Rm = w[:, l] |> w2R
				Rp = w[:, l-s] |> w2R
				R = 0.5*(Rm+Rp)
				Lm = w[:, l] |> w2L
				Lp = w[:, l-s] |> w2L
				L = 0.5*(Lm+Lp)
				for j = 1:3
					a = s*λ[i]*Rm[k, i]*Lm[i, j]
					Σ += s*Λ[i]*R[k, i]*L[i, j]*(w[j, l] - w[j, l-s])
					Σ += 0.5a* (1 - a*C) *
					( minmod(w[j, l]-w[j, l-1], w[j, l+1]-w[j,l]) -
					 minmod(w[j,l-s]-w[j,l-s-1], w[j,l-s+1]-w[j,l-s]) )
				end
			end
			wp[k, l] =w[k, l] - C*Σ
		end
	end
end

# %%

function init_non(x::AbstractVector, u::Matrix)
	w = [0.445, 0.311, 8.928]
	u[:, x .< 0] .= w2U(w)
	w = [0.5, 0, 1.4275]
	u[:, x .>= 0 ] .= w2U(w)
end

function init(x::AbstractVector, w::Matrix)
	w[:, x .< 0] .= [0.445, 0.311, 8.928]
	w[:, x .>= 0 ] .= [0.5, 0, 1.4275]
end

function true_sol(x::AbstractVector, w::Matrix, t::AbstractFloat)
	a = -2.633*t
	b = -1.636*t
	c = 1.529*t
	d = 2.480*t
	y1=[0.445, 0.311, 8.928]
	y2=[0.345, 0.527, 6.570]
	w[:, x .< a] .= y1

	k=(y1-y2)./(a-b) # y = k(x-x1)+y1
	mask=@. a < x < b
	for i = 1:3
		w[i, mask] .= k[i]*(x[mask].-a) .+ y1[i]
	end

	w[:, b.< x .< c] .= y2
	w[:, c.< x .< d] .= [1.304, 1.994, 7.691]
	w[:, x .> d] .= [0.500, 0.000, 1.428]
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

function f2title(f::Function)
	if f == upwind
		return "Upwind"
	end
	if f == limiter
		return "TVD"
	end
	if f == lax_wendroff
		return "Lax-Wendroff"
	end
	return "unknown"
end


# %%
C = 0.5
Δx= 0.01
# C = Δt/Δx
Δt =  C * Δx

# %%
function problem(C::AbstractFloat, f::Function, nx::Int = 261)

	# title = L"$m$"
	# t=0.002
	C_str=string(round(C, digits=3))
	t=0.14
	C = C/4.694
	# C = 0.7/2.633
	Δx= 2/nx
	Δt = Δx * C
	# f = limiter
	c=Cells(step=Δx, init=init)
	title = f |> f2title
	fig, ax=plt.subplots(3,1, figsize=(12,13))

	fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title, fontsize=16)
	ax[1].plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label=L"$\rho$(初始值)")
	ax[3].plot(c.x, c.u[2, :], "-.k", linewidth=0.2, label=L"$m$(初始值)")
	ax[2].plot(c.x, c.u[3, :], "-.k", linewidth=0.2, label=L"$E$(初始值)")

	flg=true # flag
	# for _ = 1:round(Int, t/Δt)
	for _ = 1:round(Int, 15)
		flg=update!(c, flg, f, C)
	end

	w=current(c, flg)
	tw=similar(w)
	true_sol(c.x, tw, t)


	ax[1].plot(c.x, tw[1, :], linewidth=1, color="k", label=L"$\rho$(真实解)", alpha=0.5)
	ax[1].plot(c.x, w[1, :], "--b", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$\rho$(数值解)")
	ax[1].set_title("密度", fontsize=14)
	ax[1].legend()
	ax[3].plot(c.x, tw[2, :], linewidth=1, color="k", label=L"$m$(真实解)", alpha=0.5)
	ax[3].plot(c.x, w[2, :], "--r", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$m$(数值解)")
	ax[2].set_title("质量流", fontsize=14)
	ax[3].legend()
	ax[2].plot(c.x, tw[3, :], linewidth=1, color="k", label=L"$E$(真实解)", alpha=0.5)
	ax[2].plot(c.x, w[3, :], "--y", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$E$(数值解)")
	ax[3].set_title("能量", fontsize=14)
	ax[2].legend()

	# plot(c.x, circshift(w, (0, 3)), tw)

	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# w = U |> U2w

	# plt.plot(x, w[1, :], linewidth=1, color="b", label="Density")
	# plt.plot(x, w[2, :], linewidth=1, color="r", label="m")
	# plt.plot(x, w[3, :], linewidth=1, color="y", label="E")
	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# plt.show()

	# plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title )
	# # plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.savefig("../figures/"*string(f)*string(nx)*".pdf", bbox_inches="tight")
	# plt.show()

end
# %%
function problem_non(C::AbstractFloat, f::Function, nx::Int = 261)

	# title = L"$m$"
	# t=0.002
	C_str=string(round(C, digits=3))
	t=0.28
	C = C/4.694
	# C = 0.7/2.633
	Δx= 2/nx
	Δt = Δx * C
	# f = limiter
	c=Cells(step=Δx, init=init_non)
	title = f |> f2title
	fig, ax=plt.subplots(3,1, figsize=(12,13))
	fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title, fontsize=16)
	ax[1].plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label=L"$\rho$(初始值)")
	ax[3].plot(c.x, c.u[2, :], "-.k", linewidth=0.2, label=L"$m$(初始值)")
	ax[2].plot(c.x, c.u[3, :], "-.k", linewidth=0.2, label=L"$E$(初始值)")

	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update!(c, flg, f, C)
	end
	U=current(c, flg)
	w = U |> U2w
	tw=similar(w)
	true_sol(c.x, tw, t)


	ax[1].plot(c.x, tw[1, :], linewidth=1, color="k", label=L"$\rho$(真实解)", alpha=0.5)
	ax[1].plot(c.x, w[1, :], "--b", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$\rho$(数值解)")
	ax[1].set_title("密度", fontsize=14)
	ax[1].legend()
	ax[3].plot(c.x, tw[2, :], linewidth=1, color="k", label=L"$m$(真实解)", alpha=0.5)
	ax[3].plot(c.x, w[2, :], "--r", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$m$(数值解)")
	ax[2].set_title("质量流", fontsize=14)
	ax[3].legend()
	ax[2].plot(c.x, tw[3, :], linewidth=1, color="k", label=L"$E$(真实解)", alpha=0.5)
	ax[2].plot(c.x, w[3, :], "--y", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$E$(数值解)")
	ax[3].set_title("能量", fontsize=14)
	ax[2].legend()

	# plot(c.x, circshift(w, (0, 3)), tw)

	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# w = U |> U2w

	# plt.plot(x, w[1, :], linewidth=1, color="b", label="Density")
	# plt.plot(x, w[2, :], linewidth=1, color="r", label="m")
	# plt.plot(x, w[3, :], linewidth=1, color="y", label="E")
	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# plt.show()

	# plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title )
	# # plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.savefig("../figures/"*string(f)*string(nx)*".pdf", bbox_inches="tight")
	# plt.show()

end
# %%

problem(0.18, limiter)
# problem1(0.18, limiter)
plt.show()

problem(0.05, limiter, 133)
plt.show()

problem(0.5, lax_wendroff)
plt.show()

problem(0.18, NND)
plt.show()


problem_non(0.5, upwind_non)
plt.show()

# problem1(0.5, upwind)

problem(0.5, upwind)
plt.show()

# %%

function main()
	problem(0.05, limiter)
	# problem1(0.18, limiter)
	problem(0.05, limiter, 133)
	problem(0.5, lax_wendroff)
	problem_non(0.1, upwind_non)
	problem(0.5, upwind)
	plt.show()
end
main()
