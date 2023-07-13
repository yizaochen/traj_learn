module Potential  
    export harmonic_well, double_well, force_double_well, force_harmonic_well, harmonic_well_k_mean, force_harmonic_well_k_mean, doulbe_well_width_height, force_doulbe_well_width_height, triple_well, symmetry_wall_potential
    
    export get_peq, get_peq_c, get_rhoeq, gaussian, get_sigma_by_k, get_k_by_sigma

    function get_sigma_by_k(k)
        return 1 / sqrt(2 * k)
    end

    function get_k_by_sigma(sigma)
        return ((1. / sigma)^2) / 2.
    end

    function get_peq(V)
        p_eq = exp.(-V);
        p_eq = p_eq ./ sum(p_eq);
        return p_eq
    end
    
    
    function get_peq_c(Vref, w0)
        peq = exp.(-Vref)
        peq = max.(peq,1e-10)
        sum_factor = sum(w0 .* peq)
        c = 1 / sum_factor
        peq = peq ./ sum_factor
        return peq, c
    end
    
    
    function get_rhoeq(Vref, w0)
        peq, c = get_peq_c(Vref, w0)
        return sqrt.(peq)
    end

    function gaussian(x::Array{Float64,1}, μ::Float64, σ::Float64)
        prefactor = (σ * √(2π))^(-1)
        inner_array = ((x .- μ) ./ σ) .^ 2
        inner_array = -0.5 * inner_array
        exp_term = exp.(inner_array)
        return prefactor * exp_term
    end

    function gaussian(x::Float64, μ::Float64, σ::Float64)
        prefactor = (σ * √(2π))^(-1)
        inner_array = ((x - μ) ./ σ) .^ 2
        inner_array = -0.5 * inner_array
        exp_term = exp.(inner_array)
        return prefactor * exp_term
    end

    function symmetry_wall_potential(left_x::Float64, right_x::Float64, sigma::Float64, scalefactor::Float64, xvalue::Float64)
        if xvalue < 0.
            return scalefactor * gaussian(xvalue, left_x, sigma)
        else
            return scalefactor * gaussian(xvalue, right_x, sigma)
        end
    end

    function symmetry_wall_potential(left_x::Float64, right_x::Float64, sigma::Float64, scalefactor::Float64, xvalue::Float64, center_value::Float64)
        if xvalue < center_value
            return scalefactor * gaussian(xvalue, left_x, sigma)
        else
            return scalefactor * gaussian(xvalue, right_x, sigma)
        end
    end

    function harmonic_well(xref, k)
        V = k * (xref .- 1) .^ 2
        return V
    end

    function harmonic_well_k_mean(xref, k, xmean)
        V = k * (xref .- xmean) .^ 2
        return V
    end

    function double_well(xref, xmean)
        inner_term = xref .- xmean
        V = (17302.265 .* inner_term .^ 4) .- (611.347 .* inner_term .^ 2) .+ 5.3992
        return V
    end

    function doulbe_well_width_height(xref, W, H, xmean, d)
        """
        xref: domain of x
        W: width
        H: height(depth)
        """
        H = -H  # Depth is negative
        
        # Calculate c, W_factor and h by given W and H
        c = sqrt(-2 * H / W^4)
        W_factor = W * sqrt(c)
        h = sqrt(W_factor * 2 * sqrt(c))
        
        # Scaling and y-intercept factors
        s = 1 # scale factor
        V = s .* (((-1/4) * h^4 .* (xref .- xmean).^2 + (1/2) * c^2 .* ( xref .- xmean).^4) .+ d)
        return V
    end

    function triple_well(xref, xmean)
        factor1 = 1. / 15.47
        inner_term = 13.9 * (xref .- xmean)
        V = factor1 * ((inner_term .^ 6) .- (15 * (inner_term .^ 4)) .+ (53 * (inner_term .^ 2)) .+ (2 * inner_term) .- 15.)
        V = V .+ 2.9266
        return V
    end

    function force_harmonic_well(x, k)
        F = 2 * k * (x - 1)
        F = -F
        return F
    end

    function force_harmonic_well_k_mean(xref, k, xmean)
        F = k * (xref .- xmean)
        F = -2 * F
        return F
    end

    function force_double_well(x, xmean)
        F = (69209.06 * (x - xmean)^3) - (1222.694 * (x - xmean))
        F = -F
        return F
    end

    function force_doulbe_well_width_height(xref::Array{Float64,2}, W, H, xmean)
        H = -H  # Depth is negative
        
        # Calculate c, W_factor and h by given W and H
        c = sqrt(-2 * H / W^4)
        W_factor = W * sqrt(c)
        h = sqrt(W_factor * 2 * sqrt(c))

        # Scaling and y-intercept factors
        s = 1 # scale factor
        F = s .* ((1/2) * h^4 .* (xref .- xmean) - 2 * c^2 .* (xref .- xmean).^3)
        return F        
    end

    function force_doulbe_well_width_height(xref::Float64, W, H, xmean)
        H = -H  # Depth is negative
        
        # Calculate c, W_factor and h by given W and H
        c = sqrt(-2 * H / W^4)
        W_factor = W * sqrt(c)
        h = sqrt(W_factor * 2 * sqrt(c))

        # Scaling and y-intercept factors
        s = 1. # scale factor
        F = s * ((1/2) * h^4 * (xref - xmean) - 2 * c^2 * (xref - xmean)^3)
        return F        
    end

 end
