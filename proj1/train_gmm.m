
classdef train_gmm
    methods

        function model = getModel(obj, allImages, orange_pixels)
            disp("running getModel");
%             model = "running getModel";
            k = 3;

            pi_i = cell(k,1);
            mu_i = cell(k,1);
            cov_i = cell(k,1);
            j = k; % number of clusters
            i = length(orange_pixels); % number of orange pixels
            a = cell(i,j);

            % convergence criteria
            e = 1.0;

            R = orange_pixels(:,1); 
            G = orange_pixels(:,2); 
            B = orange_pixels(:,3);
            
            
            % means
            r_mu = mean2(R);
            g_mu = mean2(G);
            b_mu = mean2(B);
            
            mu = [r_mu, g_mu, b_mu];
            
            % Finding covariance
            covariance = [0 0 0; 0 0 0; 0 0 0];
            for i=1:length(allImages)
                img = allImages{i};
                
                for row=1:size(R,1)
                    for col=1:size(R,2)
                            size(img);
                            x = [R(row,col); G(row,col); B(row,col)];
                            covariance = covariance + (double(x) - double(mu))*transpose(double(x)-double(mu));
                        break
                    end
                end
            end

             % Initializing the model
            for cluster=1:k
                pi_i{cluster} = rand(1,1);
                mu_i{cluster} = [256*rand(1,1); 256*rand(1,1); 256*rand(1,1)];
                cov_i{cluster} = covariance;
            end
            
            max_iters = 15;
            step = 0;
            
            current_mean = -1000;
            previous_mean = -1000;
            convergence = 1000;
            
            while step < max_iters && (convergence > 1)
                step = step + 1;
            
                % expectation step  
                sum = 0;
                
                for cluster = 1:j % looping over all clusters
                    scale = pi_i{cluster};
                    covariance = cov_i{cluster};
                    mean = mu_i{cluster};
                    
                    pixel_count = 0;
                        for row=1:size(R,1) % looping over pixels of an image
                            for col=1:size(R,2)
                                pixel_count = pixel_count + 1;
                        
                                x = [double(R(row,col)); double(G(row,col)); double(B(row,col))]; % getting a single [r g b] value
                                p = (-1/sqrt((2*pi)^3 * det(covariance)))*exp(-0.5*transpose(x-mean)*((covariance)\(x-mean))); % calculating likelihood
                                numerator = scale * p; % numerator of alpha eqn
                                a{pixel_count, cluster} = numerator;
                                sum = sum + numerator; % this will later become the denominator
                                break                
                            end
                        end
                end
                A = cell2mat(a);
                % Divide alpha by sum
                A = A./sum; % A is alpha
            
                
                % Maximization step
                
                % Calculating mu's, covariances, and  for each cluster
                for cluster=1:k
            %         mu_i{cluster} = 0;
                    numerator_mu = 0;
                    denominator_mu = 0;
                    numerator_cov = 0;
                    
                    pixel_count = 0;
                    
                    % Populating mu_i
                    for row=1:size(R,1) % looping over pixels of an image
                        for col=1:size(R,2)
                            pixel_count = pixel_count + 1;
                            x = [double(R(row,col)); double(G(row,col)); double(B(row,col))]; % getting a single [r g b] value
            
                            numerator_mu = numerator_mu + A(pixel_count,cluster) * x;
                            denominator_mu = denominator_mu + A(pixel_count,cluster);
            
                        end
                    end
                    mu_i{cluster} = numerator_mu/denominator_mu;
            
                    % Populating cov_i
                    pixel_count = 0;
                    for row=1:size(R,1) % looping over pixels of an image
                        for col=1:size(R,2)
                            pixel_count = pixel_count + 1;
                            x = [double(R(row,col)); double(G(row,col)); double(B(row,col))]; % getting a single [r g b] value
                            
                            numerator_cov = numerator_cov + A(pixel_count, cluster)*((x - mu_i{cluster})*transpose(x - mu_i{cluster}));
                        end
                    end
            
                    cov_i{cluster} = numerator_cov/denominator_mu;
                    pi_i{cluster} = denominator_mu/pixel_count;
                            
                end
                
                means = cell2mat(mu_i);
                
             
                if step == 1
                      previous_mean = means;
                elseif step == 2
                    current_mean = means;
                    convergence = abs(norm(previous_mean - current_mean));
                else
                    previous_mean = current_mean;
                    current_mean = means;
                    convergence = abs(norm(previous_mean - current_mean));
                end
            
%                 disp(convergence);
              
            end
            model = [mu_i, cov_i, pi_i];
        end
    end
end