create table portfolio (
discount int,
informational int,
bogo int,
id string,
difficulty int,
duration int,
offer_type string,
reward int,
channel_mobile int,
channel_email int,
channel_social int,
channel_web int)
row format delimited
fields terminated by ','
tblproperties("skip.header.line.count"="1");

Load data inpath 's3://finalprojectmbd/portfolio_cleaned.csv' into table portfolio;

select * from portfolio;

create table profile (
gender_F int,
gender_M int,
gender_O int,
joined_in_2018 int,
joined_in_2015 int,
joined_in_2013 int,
joined_in_2014 int,
joined_in_2016 int,
joined_in_2017 int,
age int,
id string,
income int)
row format delimited
fields terminated by ','
tblproperties("skip.header.line.count"="1");

Load data inpath 's3://finalprojectmbd/profile_cleaned.csv' into table profile;

select * from profile;

create table transcript (
transaction int,
offer_received int,
offer_compelted int,
offer_viewed int,
event string,
person string,
time int,
amount float,
reward int,
offer_id string)
row format delimited
fields terminated by ','
tblproperties("skip.header.line.count"="1");

Load data inpath 's3://finalprojectmbd/transcript_cleaned.csv' into table transcript;

select * from transcript;

create table clustered (
person string,
gender_F int,
gender_M int,
gender_O int,
joined_in_2018 int,
joined_in_2015 int,
joined_in_2013 int,
joined_in_2014 int,
joined_in_2016 int,
joined_in_2017 int,
age int,
income int,
amount_per_visit float,
frequency int,
bogo_view_rate float,
bogo_complete_rate float,
discount_view_rate float,
discount_complete_rate float,
informational_view_rate float,
non_info_view_rate float,
non_info_complete_rate float,
reward_avg float,
difficulty_avg float,
email_cnt int,
mobile_cnt int,
social_cnt int,
web_cnt int,
total_bogo_received int,
total_discount_received int,
total_informational_received int,
received_per_offer float,
viewed_per_offer float,
completed_per_offer float,
total_offers_received int,
total_offers_viewed int,
total_offers_completed int,
prediction int)
row format delimited
fields terminated by ','
tblproperties("skip.header.line.count"="1");

Load data inpath 's3://finalprojectmbd/starbucks-clustered.csv' into table clustered;

select * from clustered;


select count(*) from transcript where transcript.offer_received=1
union all
select count(*) from transcript where transcript.offer_compelted=1 
union all
select count(*) from transcript where transcript.offer_viewed=1 
union all
select count(*) from transcript where transcript.transaction=1 
;


create table preincome as
select event, avg(income) avgincome from
(select * from transcript t left outer join profile p on p.id=t.person) combined
group by event;

create table preage as
select event, avg(age) avgage from
(select * from transcript t left outer join profile p on p.id=t.person) combined
group by event;

select * from preage;

create table prewomen as
select event, sum(gender_f) women from
(select * from transcript t left outer join profile p on p.id=t.person) combined
group by event;

create table premen as
select event, sum(gender_m) men from
(select * from transcript t left outer join profile p on p.id=t.person) combined
group by event;

create table preother as
select event, sum(gender_o) other from
(select * from transcript t left outer join profile p on p.id=t.person) combined
group by event;

select i.event, i.avgincome, a.avgage, w.women, m.men, o.other
from preincome i join preage a on i.event = a.event
join prewomen w on i.event = w.event
join premen m on i.event = m.event
join preother o on i.event=o.event;

create table avgincome as
select prediction, avg(income) avgincome from clustered group by prediction;

create table avgage as
select prediction, avg(age) avgage from clustered group by prediction;

select i.prediction, i.avgincome, a.avgage, s.women, m.men, o.other, r.avg_offers_received, c.avg_offers_completed, v.avg_offers_viewed, t.avgamountpervisist, f.avgfrequency 
from avgincome i join avgage a on i.prediction=a.prediction
join sumwomen s on i.prediction=s.prediction
join summen m on i.prediction=m.prediction
join sumother o on i.prediction=o.prediction
join received r on i.prediction=r.prediction
join completed c on i.prediction=c.prediction
join viewed v on i.prediction=v.prediction
join visit t on i.prediction=t.prediction
join frequency f on i.prediction=f.prediction
;

create table sumwomen as
select prediction, sum(gender_f) women from clustered group by prediction;
create table summen as
select prediction, sum(gender_m) men from clustered group by prediction;
create table sumother as
select prediction, sum(gender_o) other from clustered group by prediction;

create table received as
select prediction, avg(total_offers_received) avg_offers_received from clustered group by prediction;
create table completed as
select prediction, avg(total_offers_completed) avg_offers_completed from clustered group by prediction;
create table viewed as
select prediction, avg(total_offers_viewed) avg_offers_viewed from clustered group by prediction;
create table visit as
select prediction, avg(amount_per_visit) avgamountpervisist from clustered group by prediction;
create table frequency as
select prediction, avg(frequency) avgfrequency from clustered group by prediction;