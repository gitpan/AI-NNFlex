use AI::NNFlex;
use strict;

my $object = AI::NNFlex->new([{"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                        {"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                       {"nodes"=>1,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"linear","random weights"=>1}],{'random connections'=>0,'networktype'=>'feedforward', 'random weights'=>1,'learning algorithm'=>'backprop','learning rate'=>.3,'debug'=>[2,3,4],'bias'=>1});
print "Debug = ".@{$object->{'debug'}}."\n";
$object->dump_state({'filename'=>"weights.wts",'activations'=>1});
$object->{'debug'} = [];
my $err = 10;
my $counter=1;
while ($err >.001)
{
$object->run([0,0]);
$err = $object->learn([1]);
$object->run([0,1]);
$err = $object->learn([0]);
$object->run([1,0]);
$err = $object->learn([0]);
$object->run([1,1]);
$err = $object->learn([1]);
print "Epoch $counter : Error = $err\n";
$counter++;
}
print "Error = $err\n";




$object->dump_state({'filename'=>"weights-learned.wts",'activations'=>1});

#$object->{'debug'}=[3,4];

print "Debug = ".@{$object->{'debug'}}."\n";
$object->run([1,1]);
my $output = $object->output();

foreach (@$output){
print "1 1 should be 1 - ".$_."\n";}

$object->run([0,1]);
my $output = $object->output();
foreach (@$output){
print "0 1 should be 0 - ".$_."\n";}

$object->run([0,0]);
my $output = $object->output();
foreach (@$output){
print "0 0 should be 1 - ".$_."\n";}

$object->run([1,0]);
my $output = $object->output();
foreach (@$output){
print "1 0 should be 0 - ".$_."\n";}

$object->dump_state({'filename'=>"weights-run.wts",'activations'=>1});



#use AI::NNFlex::draw;

#my $image = AI::NNFlex::draw->network($object);

#open (GIF,">and.gif");
#binmode GIF;
#print GIF $image;
#close GIF;

