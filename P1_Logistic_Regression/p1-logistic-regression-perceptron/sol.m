features=table2array(diabetes(:,1:8));
labels=table2array(diabetes(:,9));
w=zeros(8,1);
b=0;
lr=.1;
for i=1:8
    mx=max(features(:,i));
    mn=min(features(:,i));
    features(:,i)=(features(:,i)-(mx+mn)/2)/((mx-mn)/2);
end
for i=1:1000
    cost(i)=0;
    for j=1:768
        %y_hat=(features(j,:)*w+b>0);
        y_hat=1/(1+exp(-(features(j,:)*w+b)));
        err=labels(j)-y_hat;
        w=w+lr*err*y_hat*(1-y_hat)*features(j,:)';
        b=b+lr*err*y_hat*(1-y_hat);
        cost(i)=cost(i)+err*err;
    end
    i
end
y_hat=1/(1+exp(-(features*w+b)))>=0.5;
%y_hat=(features*w+b>0);
sum((y_hat'==labels)/768*100)
plot(cost/768)