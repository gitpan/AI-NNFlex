# Example demonstrating XOR


use AI::NNFlex;

# Create the network 

my $object = AI::NNFlex->new([{"nodes"=>2,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                        {"nodes"=>3,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"tanh","random weights"=>1},
                       {"nodes"=>1,"persistent activation"=>0,"decay"=>0.0,"random activation"=>0,"threshold"=>0.0,"activation function"=>"linear","random weights"=>1}],
{'random connections'=>0,'networktype'=>'feedforward', 'random weights'=>1,'learning algorithm'=>'backprop','learning rate'=>.3,'debug'=>[],'bias'=>1});

# dump the current state of the network to a file
$object->dump_state({'filename'=>"weights.wts",'activations'=>1});


my $err = 10;
my $counter=1;
while ($err >.01)
{
# run a pattern through the network
$object->run([0,0]);

# adjust weights using backprop (see definition of net above)
$err = $object->learn([0]);

$object->run([0,1]);
$err = $object->learn([1]);

$object->run([1,1]);
$err = $object->learn([0]);

$object->run([1,0]);
$err = $object->learn([1]);

print "Epoch $counter : Error = $err\n";
$counter++;
}
print "Error = $err\n";




$object->dump_state({'filename'=>"weights-learned.wts",'activations'=>1});

# change the debug level
$object->{'debug'}=[4];

# run a pattern through the network
$object->run([1,1]);

# get the resulting activation on the output layer
my $output = $object->output();

foreach (@$output){
print "1 1 should be 0 - ".$_."\n";}

$object->run([0,1]);
my $output = $object->output();
foreach (@$output){
print "0 1 should be 1 - ".$_."\n";}

$object->run([0,0]);
my $output = $object->output();
foreach (@$output){
print "0 0 should be 0 - ".$_."\n";}

$object->run([1,0]);
my $output = $object->output();
foreach (@$output){
print "1 0 should be 1 - ".$_."\n";}

$object->dump_state({'filename'=>"weights-run.wts",'activations'=>1});


# use the experimental network drawing package
use AI::NNFlex::draw;

# draw a diagram of the network
my $image = AI::NNFlex::draw->network($object);

# and write it out to a file
open (GIF,">xor.gif");
binmode GIF;
print GIF $image;
close GIF;

