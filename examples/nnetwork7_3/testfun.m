function testfun(x, y)

    tic;
    figure
    hold on 
    length(x)
    for i=1:length(x)
        Vs = x{i};
        Vs_tem = [];
        for j =1:length(Vs)
            P = Vs{j};
            p_temp = [];
            for n = 1:length(P)
                p_temp = [p_temp, P{n}];
            end 
            Vs_tem = [Vs_tem; p_temp];
        end  
        poly0 = Polyhedron(Vs_tem);
        plot(poly0)
    end
    for i = 1:length(y)
        plot(y(i,1), y(i,2), '*b')
    end
  
    hold off
    savefig('fig')
    toc
end

