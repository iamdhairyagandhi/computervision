classdef plot_gmm
    methods

        function grapher = plot_ellipsoid(obj, model)
            for i = 1:size(model)
                disp(i)
                f = figure
                error_ellipse(model{i,2})
                %plot(error_ellipse(model{i,2}))
            end
        end
    end
end



% plotter = plot_gmm;
% plotter.plot_ellipsoid(model)
