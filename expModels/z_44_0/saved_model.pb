г┌
Лх
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12unknown8Б╔
ё
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
: *
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
: *
dtype0
ё
conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_61/kernel
}
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_61/bias
m
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes
:@*
dtype0
|
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_50/kernel
u
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_50/bias
l
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes	
:ђ*
dtype0
{
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђX* 
shared_namedense_51/kernel
t
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes
:	ђX*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:X*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:╚*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:╚*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:╚*
dtype0
њ
Adam/conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_60/kernel/m
І
+Adam/conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_60/bias/m
{
)Adam/conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/m*
_output_shapes
: *
dtype0
њ
Adam/conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_61/kernel/m
І
+Adam/conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/m*&
_output_shapes
: @*
dtype0
ѓ
Adam/conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_61/bias/m
{
)Adam/conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/m*
_output_shapes
:@*
dtype0
і
Adam/dense_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_50/kernel/m
Ѓ
*Adam/dense_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_50/bias/m
z
(Adam/dense_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђX*'
shared_nameAdam/dense_51/kernel/m
ѓ
*Adam/dense_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/m*
_output_shapes
:	ђX*
dtype0
ђ
Adam/dense_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_51/bias/m
y
(Adam/dense_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/m*
_output_shapes
:X*
dtype0
њ
Adam/conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_60/kernel/v
І
+Adam/conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_60/bias/v
{
)Adam/conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/v*
_output_shapes
: *
dtype0
њ
Adam/conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_61/kernel/v
І
+Adam/conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/v*&
_output_shapes
: @*
dtype0
ѓ
Adam/conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_61/bias/v
{
)Adam/conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/v*
_output_shapes
:@*
dtype0
і
Adam/dense_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_50/kernel/v
Ѓ
*Adam/dense_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_50/bias/v
z
(Adam/dense_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђX*'
shared_nameAdam/dense_51/kernel/v
ѓ
*Adam/dense_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/v*
_output_shapes
:	ђX*
dtype0
ђ
Adam/dense_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_51/bias/v
y
(Adam/dense_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/v*
_output_shapes
:X*
dtype0

NoOpNoOp
ьC
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*еC
valueъCBЏC BћC
█
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
Я
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemЋmќmЌ mў1mЎ2mџ;mЏ<mюvЮvъvЪ vа1vА2vб;vБ<vц
8
0
1
2
 3
14
25
;6
<7
8
0
1
2
 3
14
25
;6
<7
 
Г
Fmetrics

Glayers
trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
	variables
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г
Kmetrics
	variables

Llayers
trainable_variables
Mlayer_regularization_losses
Nnon_trainable_variables
Olayer_metrics
regularization_losses
 
 
 
Г
Pmetrics
	variables

Qlayers
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics
regularization_losses
 
 
 
Г
Umetrics
	variables

Vlayers
trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
Ylayer_metrics
regularization_losses
\Z
VARIABLE_VALUEconv2d_61/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_61/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
Г
Zmetrics
!	variables

[layers
"trainable_variables
\layer_regularization_losses
]non_trainable_variables
^layer_metrics
#regularization_losses
 
 
 
Г
_metrics
%	variables

`layers
&trainable_variables
alayer_regularization_losses
bnon_trainable_variables
clayer_metrics
'regularization_losses
 
 
 
Г
dmetrics
)	variables

elayers
*trainable_variables
flayer_regularization_losses
gnon_trainable_variables
hlayer_metrics
+regularization_losses
 
 
 
Г
imetrics
-	variables

jlayers
.trainable_variables
klayer_regularization_losses
lnon_trainable_variables
mlayer_metrics
/regularization_losses
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
Г
nmetrics
3	variables

olayers
4trainable_variables
player_regularization_losses
qnon_trainable_variables
rlayer_metrics
5regularization_losses
 
 
 
Г
smetrics
7	variables

tlayers
8trainable_variables
ulayer_regularization_losses
vnon_trainable_variables
wlayer_metrics
9regularization_losses
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
Г
xmetrics
=	variables

ylayers
>trainable_variables
zlayer_regularization_losses
{non_trainable_variables
|layer_metrics
?regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

}0
~1
2
ђ3
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Ђtotal

ѓcount
Ѓ	variables
ё	keras_api
\
Ё
thresholds
єtrue_positives
Єfalse_positives
ѕ	variables
Ѕ	keras_api
\
і
thresholds
Іtrue_positives
їfalse_negatives
Ї	variables
ј	keras_api
v
Јtrue_positives
љtrue_negatives
Љfalse_positives
њfalse_negatives
Њ	variables
ћ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ђ0
ѓ1

Ѓ	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE

є0
Є1

ѕ	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

І0
ї1

Ї	variables
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
Ј0
љ1
Љ2
њ3

Њ	variables
}
VARIABLE_VALUEAdam/conv2d_60/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_50/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_50/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_60/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_50/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_50/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ћ
serving_default_conv2d_60_inputPlaceholder*0
_output_shapes
:         └*
dtype0*%
shape:         └
н
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_60_inputconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *0
f+R)
'__inference_signature_wrapper_133611531
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
»
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Adam/conv2d_60/kernel/m/Read/ReadVariableOp)Adam/conv2d_60/bias/m/Read/ReadVariableOp+Adam/conv2d_61/kernel/m/Read/ReadVariableOp)Adam/conv2d_61/bias/m/Read/ReadVariableOp*Adam/dense_50/kernel/m/Read/ReadVariableOp(Adam/dense_50/bias/m/Read/ReadVariableOp*Adam/dense_51/kernel/m/Read/ReadVariableOp(Adam/dense_51/bias/m/Read/ReadVariableOp+Adam/conv2d_60/kernel/v/Read/ReadVariableOp)Adam/conv2d_60/bias/v/Read/ReadVariableOp+Adam/conv2d_61/kernel/v/Read/ReadVariableOp)Adam/conv2d_61/bias/v/Read/ReadVariableOp*Adam/dense_50/kernel/v/Read/ReadVariableOp(Adam/dense_50/bias/v/Read/ReadVariableOp*Adam/dense_51/kernel/v/Read/ReadVariableOp(Adam/dense_51/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__traced_save_133611984
ъ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1Adam/conv2d_60/kernel/mAdam/conv2d_60/bias/mAdam/conv2d_61/kernel/mAdam/conv2d_61/bias/mAdam/dense_50/kernel/mAdam/dense_50/bias/mAdam/dense_51/kernel/mAdam/dense_51/bias/mAdam/conv2d_60/kernel/vAdam/conv2d_60/bias/vAdam/conv2d_61/kernel/vAdam/conv2d_61/bias/vAdam/dense_50/kernel/vAdam/dense_50/bias/vAdam/dense_51/kernel/vAdam/dense_51/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference__traced_restore_133612111иъ
г*
й
C__inference_m_44_layer_call_and_return_conditional_losses_133611397
conv2d_60_input
conv2d_60_133611370
conv2d_60_133611372
conv2d_61_133611377
conv2d_61_133611379
dense_50_133611385
dense_50_133611387
dense_51_133611391
dense_51_133611393
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб dense_50/StatefulPartitionedCallб dense_51/StatefulPartitionedCallи
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallconv2d_60_inputconv2d_60_133611370conv2d_60_133611372*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_60_layer_call_and_return_conditional_losses_1336111632#
!conv2d_60/StatefulPartitionedCallЇ
dropout_65/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111962
dropout_65/PartitionedCallЌ
 max_pooling2d_60/PartitionedCallPartitionedCall#dropout_65/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_1336111302"
 max_pooling2d_60/PartitionedCallл
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0conv2d_61_133611377conv2d_61_133611379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_61_layer_call_and_return_conditional_losses_1336112212#
!conv2d_61/StatefulPartitionedCallї
dropout_66/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112542
dropout_66/PartitionedCallЌ
 max_pooling2d_61/PartitionedCallPartitionedCall#dropout_66/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_1336111422"
 max_pooling2d_61/PartitionedCallё
flatten_40/PartitionedCallPartitionedCall)max_pooling2d_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flatten_40_layer_call_and_return_conditional_losses_1336112742
flatten_40/PartitionedCallЙ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_50_133611385dense_50_133611387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_50_layer_call_and_return_conditional_losses_1336112932"
 dense_50/StatefulPartitionedCallё
dropout_67/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113262
dropout_67/PartitionedCallй
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_67/PartitionedCall:output:0dense_51_133611391dense_51_133611393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_51_layer_call_and_return_conditional_losses_1336113502"
 dense_51/StatefulPartitionedCallІ
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
В
g
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611254

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         I@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         I@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         I@:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
¤
h
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611704

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         Е 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         Е *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         Е 2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         Е 2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         Е 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Е :X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
░
О
(__inference_m_44_layer_call_fn_133611651

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_m_44_layer_call_and_return_conditional_losses_1336114302
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
­
g
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611196

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         Е 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         Е 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         Е :X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
Ш	
Я
G__inference_dense_51_layer_call_and_return_conditional_losses_133611835

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђX*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         X2	
Sigmoidљ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Щ	
Я
G__inference_dense_50_layer_call_and_return_conditional_losses_133611293

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
л
g
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611326

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ*
┤
C__inference_m_44_layer_call_and_return_conditional_losses_133611481

inputs
conv2d_60_133611454
conv2d_60_133611456
conv2d_61_133611461
conv2d_61_133611463
dense_50_133611469
dense_50_133611471
dense_51_133611475
dense_51_133611477
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб dense_50/StatefulPartitionedCallб dense_51/StatefulPartitionedCall«
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_133611454conv2d_60_133611456*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_60_layer_call_and_return_conditional_losses_1336111632#
!conv2d_60/StatefulPartitionedCallЇ
dropout_65/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111962
dropout_65/PartitionedCallЌ
 max_pooling2d_60/PartitionedCallPartitionedCall#dropout_65/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_1336111302"
 max_pooling2d_60/PartitionedCallл
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0conv2d_61_133611461conv2d_61_133611463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_61_layer_call_and_return_conditional_losses_1336112212#
!conv2d_61/StatefulPartitionedCallї
dropout_66/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112542
dropout_66/PartitionedCallЌ
 max_pooling2d_61/PartitionedCallPartitionedCall#dropout_66/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_1336111422"
 max_pooling2d_61/PartitionedCallё
flatten_40/PartitionedCallPartitionedCall)max_pooling2d_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flatten_40_layer_call_and_return_conditional_losses_1336112742
flatten_40/PartitionedCallЙ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_50_133611469dense_50_133611471*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_50_layer_call_and_return_conditional_losses_1336112932"
 dense_50/StatefulPartitionedCallё
dropout_67/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113262
dropout_67/PartitionedCallй
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_67/PartitionedCall:output:0dense_51_133611475dense_51_133611477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_51_layer_call_and_return_conditional_losses_1336113502"
 dense_51/StatefulPartitionedCallІ
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
╠N
Р
C__inference_m_44_layer_call_and_return_conditional_losses_133611591

inputs,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource,
(conv2d_61_conv2d_readvariableop_resource-
)conv2d_61_biasadd_readvariableop_resource+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource
identityѕб conv2d_60/BiasAdd/ReadVariableOpбconv2d_60/Conv2D/ReadVariableOpб conv2d_61/BiasAdd/ReadVariableOpбconv2d_61/Conv2D/ReadVariableOpбdense_50/BiasAdd/ReadVariableOpбdense_50/MatMul/ReadVariableOpбdense_51/BiasAdd/ReadVariableOpбdense_51/MatMul/ReadVariableOp│
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_60/Conv2D/ReadVariableOp├
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е *
paddingVALID*
strides
2
conv2d_60/Conv2Dф
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp▒
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е 2
conv2d_60/BiasAdd
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:         Е 2
conv2d_60/Reluy
dropout_65/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_65/dropout/Const│
dropout_65/dropout/MulMulconv2d_60/Relu:activations:0!dropout_65/dropout/Const:output:0*
T0*0
_output_shapes
:         Е 2
dropout_65/dropout/Mulђ
dropout_65/dropout/ShapeShapeconv2d_60/Relu:activations:0*
T0*
_output_shapes
:2
dropout_65/dropout/Shapeя
/dropout_65/dropout/random_uniform/RandomUniformRandomUniform!dropout_65/dropout/Shape:output:0*
T0*0
_output_shapes
:         Е *
dtype021
/dropout_65/dropout/random_uniform/RandomUniformІ
!dropout_65/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_65/dropout/GreaterEqual/yз
dropout_65/dropout/GreaterEqualGreaterEqual8dropout_65/dropout/random_uniform/RandomUniform:output:0*dropout_65/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         Е 2!
dropout_65/dropout/GreaterEqualЕ
dropout_65/dropout/CastCast#dropout_65/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         Е 2
dropout_65/dropout/Cast»
dropout_65/dropout/Mul_1Muldropout_65/dropout/Mul:z:0dropout_65/dropout/Cast:y:0*
T0*0
_output_shapes
:         Е 2
dropout_65/dropout/Mul_1╩
max_pooling2d_60/MaxPoolMaxPooldropout_65/dropout/Mul_1:z:0*/
_output_shapes
:         T *
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPool│
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_61/Conv2D/ReadVariableOpП
conv2d_61/Conv2DConv2D!max_pooling2d_60/MaxPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@*
paddingVALID*
strides
2
conv2d_61/Conv2Dф
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp░
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@2
conv2d_61/BiasAdd~
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:         I@2
conv2d_61/Reluy
dropout_66/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_66/dropout/Const▓
dropout_66/dropout/MulMulconv2d_61/Relu:activations:0!dropout_66/dropout/Const:output:0*
T0*/
_output_shapes
:         I@2
dropout_66/dropout/Mulђ
dropout_66/dropout/ShapeShapeconv2d_61/Relu:activations:0*
T0*
_output_shapes
:2
dropout_66/dropout/ShapeП
/dropout_66/dropout/random_uniform/RandomUniformRandomUniform!dropout_66/dropout/Shape:output:0*
T0*/
_output_shapes
:         I@*
dtype021
/dropout_66/dropout/random_uniform/RandomUniformІ
!dropout_66/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_66/dropout/GreaterEqual/yЫ
dropout_66/dropout/GreaterEqualGreaterEqual8dropout_66/dropout/random_uniform/RandomUniform:output:0*dropout_66/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         I@2!
dropout_66/dropout/GreaterEqualе
dropout_66/dropout/CastCast#dropout_66/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         I@2
dropout_66/dropout/Cast«
dropout_66/dropout/Mul_1Muldropout_66/dropout/Mul:z:0dropout_66/dropout/Cast:y:0*
T0*/
_output_shapes
:         I@2
dropout_66/dropout/Mul_1╩
max_pooling2d_61/MaxPoolMaxPooldropout_66/dropout/Mul_1:z:0*/
_output_shapes
:         $@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_61/MaxPoolu
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
flatten_40/Constц
flatten_40/ReshapeReshape!max_pooling2d_61/MaxPool:output:0flatten_40/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_40/Reshapeф
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_50/MatMul/ReadVariableOpц
dense_50/MatMulMatMulflatten_40/Reshape:output:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_50/MatMulе
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_50/BiasAdd/ReadVariableOpд
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_50/BiasAddt
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_50/Reluy
dropout_67/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_67/dropout/Constф
dropout_67/dropout/MulMuldense_50/Relu:activations:0!dropout_67/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_67/dropout/Mul
dropout_67/dropout/ShapeShapedense_50/Relu:activations:0*
T0*
_output_shapes
:2
dropout_67/dropout/Shapeо
/dropout_67/dropout/random_uniform/RandomUniformRandomUniform!dropout_67/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_67/dropout/random_uniform/RandomUniformІ
!dropout_67/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_67/dropout/GreaterEqual/yв
dropout_67/dropout/GreaterEqualGreaterEqual8dropout_67/dropout/random_uniform/RandomUniform:output:0*dropout_67/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_67/dropout/GreaterEqualА
dropout_67/dropout/CastCast#dropout_67/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_67/dropout/CastД
dropout_67/dropout/Mul_1Muldropout_67/dropout/Mul:z:0dropout_67/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_67/dropout/Mul_1Е
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes
:	ђX*
dtype02 
dense_51/MatMul/ReadVariableOpц
dense_51/MatMulMatMuldropout_67/dropout/Mul_1:z:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_51/MatMulД
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_51/BiasAdd/ReadVariableOpЦ
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_51/BiasAdd|
dense_51/SigmoidSigmoiddense_51/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_51/SigmoidЭ
IdentityIdentitydense_51/Sigmoid:y:0!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
­
g
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611709

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         Е 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         Е 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         Е :X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
ї
ѓ
-__inference_conv2d_60_layer_call_fn_133611692

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_60_layer_call_and_return_conditional_losses_1336111632
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
ј
h
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611321

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
И
P
4__inference_max_pooling2d_60_layer_call_fn_133611136

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_1336111302
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┴
e
I__inference_flatten_40_layer_call_and_return_conditional_losses_133611274

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         $@:W S
/
_output_shapes
:         $@
 
_user_specified_nameinputs
э.
Б
C__inference_m_44_layer_call_and_return_conditional_losses_133611430

inputs
conv2d_60_133611403
conv2d_60_133611405
conv2d_61_133611410
conv2d_61_133611412
dense_50_133611418
dense_50_133611420
dense_51_133611424
dense_51_133611426
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб dense_50/StatefulPartitionedCallб dense_51/StatefulPartitionedCallб"dropout_65/StatefulPartitionedCallб"dropout_66/StatefulPartitionedCallб"dropout_67/StatefulPartitionedCall«
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_133611403conv2d_60_133611405*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_60_layer_call_and_return_conditional_losses_1336111632#
!conv2d_60/StatefulPartitionedCallЦ
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111912$
"dropout_65/StatefulPartitionedCallЪ
 max_pooling2d_60/PartitionedCallPartitionedCall+dropout_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_1336111302"
 max_pooling2d_60/PartitionedCallл
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0conv2d_61_133611410conv2d_61_133611412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_61_layer_call_and_return_conditional_losses_1336112212#
!conv2d_61/StatefulPartitionedCall╔
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112492$
"dropout_66/StatefulPartitionedCallЪ
 max_pooling2d_61/PartitionedCallPartitionedCall+dropout_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_1336111422"
 max_pooling2d_61/PartitionedCallё
flatten_40/PartitionedCallPartitionedCall)max_pooling2d_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flatten_40_layer_call_and_return_conditional_losses_1336112742
flatten_40/PartitionedCallЙ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_50_133611418dense_50_133611420*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_50_layer_call_and_return_conditional_losses_1336112932"
 dense_50/StatefulPartitionedCall┴
"dropout_67/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0#^dropout_66/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113212$
"dropout_67/StatefulPartitionedCall┼
 dense_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_67/StatefulPartitionedCall:output:0dense_51_133611424dense_51_133611426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_51_layer_call_and_return_conditional_losses_1336113502"
 dense_51/StatefulPartitionedCallЩ
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2H
"dropout_67/StatefulPartitionedCall"dropout_67/StatefulPartitionedCall:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
┘

р
H__inference_conv2d_60_layer_call_and_return_conditional_losses_133611683

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Е 2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
И
P
4__inference_max_pooling2d_61_layer_call_fn_133611148

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_1336111422
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
g
.__inference_dropout_66_layer_call_fn_133611761

inputs
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112492
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         I@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
М

р
H__inference_conv2d_61_layer_call_and_return_conditional_losses_133611730

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         I@2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         T ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         T 
 
_user_specified_nameinputs
Ё
k
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_133611142

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
К
h
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611751

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         I@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         I@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         I@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         I@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         I@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         I@:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
┐
J
.__inference_dropout_66_layer_call_fn_133611766

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         I@:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
»
g
.__inference_dropout_67_layer_call_fn_133611819

inputs
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113212
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Б
J
.__inference_dropout_67_layer_call_fn_133611824

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
К
h
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611249

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         I@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         I@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         I@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         I@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         I@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         I@:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
Ш	
Я
G__inference_dense_51_layer_call_and_return_conditional_losses_133611350

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђX*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         X2	
Sigmoidљ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
У
Ђ
,__inference_dense_51_layer_call_fn_133611844

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_51_layer_call_and_return_conditional_losses_1336113502
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
░
О
(__inference_m_44_layer_call_fn_133611672

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_m_44_layer_call_and_return_conditional_losses_1336114812
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
╦
Я
(__inference_m_44_layer_call_fn_133611500
conv2d_60_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallconv2d_60_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_m_44_layer_call_and_return_conditional_losses_1336114812
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
Ж
Ђ
,__inference_dense_50_layer_call_fn_133611797

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_50_layer_call_and_return_conditional_losses_1336112932
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
НR
Ѓ
"__inference__traced_save_133611984
file_prefix/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_adam_conv2d_60_kernel_m_read_readvariableop4
0savev2_adam_conv2d_60_bias_m_read_readvariableop6
2savev2_adam_conv2d_61_kernel_m_read_readvariableop4
0savev2_adam_conv2d_61_bias_m_read_readvariableop5
1savev2_adam_dense_50_kernel_m_read_readvariableop3
/savev2_adam_dense_50_bias_m_read_readvariableop5
1savev2_adam_dense_51_kernel_m_read_readvariableop3
/savev2_adam_dense_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_60_kernel_v_read_readvariableop4
0savev2_adam_conv2d_60_bias_v_read_readvariableop6
2savev2_adam_conv2d_61_kernel_v_read_readvariableop4
0savev2_adam_conv2d_61_bias_v_read_readvariableop5
1savev2_adam_dense_50_kernel_v_read_readvariableop3
/savev2_adam_dense_50_bias_v_read_readvariableop5
1savev2_adam_dense_51_kernel_v_read_readvariableop3
/savev2_adam_dense_51_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameо
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*У
valueяB█(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesп
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┘
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_adam_conv2d_60_kernel_m_read_readvariableop0savev2_adam_conv2d_60_bias_m_read_readvariableop2savev2_adam_conv2d_61_kernel_m_read_readvariableop0savev2_adam_conv2d_61_bias_m_read_readvariableop1savev2_adam_dense_50_kernel_m_read_readvariableop/savev2_adam_dense_50_bias_m_read_readvariableop1savev2_adam_dense_51_kernel_m_read_readvariableop/savev2_adam_dense_51_bias_m_read_readvariableop2savev2_adam_conv2d_60_kernel_v_read_readvariableop0savev2_adam_conv2d_60_bias_v_read_readvariableop2savev2_adam_conv2d_61_kernel_v_read_readvariableop0savev2_adam_conv2d_61_bias_v_read_readvariableop1savev2_adam_dense_50_kernel_v_read_readvariableop/savev2_adam_dense_50_bias_v_read_readvariableop1savev2_adam_dense_51_kernel_v_read_readvariableop/savev2_adam_dense_51_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*О
_input_shapes┼
┬: : : : @:@:
ђђ:ђ:	ђX:X: : : : : : : :::::╚:╚:╚:╚: : : @:@:
ђђ:ђ:	ђX:X: : : @:@:
ђђ:ђ:	ђX:X: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђX: 

_output_shapes
:X:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђX: 

_output_shapes
:X:, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:&$"
 
_output_shapes
:
ђђ:!%

_output_shapes	
:ђ:%&!

_output_shapes
:	ђX: '

_output_shapes
:X:(

_output_shapes
: 
ј
h
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611809

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
л
g
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611814

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▒ц
Љ
%__inference__traced_restore_133612111
file_prefix%
!assignvariableop_conv2d_60_kernel%
!assignvariableop_1_conv2d_60_bias'
#assignvariableop_2_conv2d_61_kernel%
!assignvariableop_3_conv2d_61_bias&
"assignvariableop_4_dense_50_kernel$
 assignvariableop_5_dense_50_bias&
"assignvariableop_6_dense_51_kernel$
 assignvariableop_7_dense_51_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count&
"assignvariableop_15_true_positives'
#assignvariableop_16_false_positives(
$assignvariableop_17_true_positives_1'
#assignvariableop_18_false_negatives(
$assignvariableop_19_true_positives_2&
"assignvariableop_20_true_negatives)
%assignvariableop_21_false_positives_1)
%assignvariableop_22_false_negatives_1/
+assignvariableop_23_adam_conv2d_60_kernel_m-
)assignvariableop_24_adam_conv2d_60_bias_m/
+assignvariableop_25_adam_conv2d_61_kernel_m-
)assignvariableop_26_adam_conv2d_61_bias_m.
*assignvariableop_27_adam_dense_50_kernel_m,
(assignvariableop_28_adam_dense_50_bias_m.
*assignvariableop_29_adam_dense_51_kernel_m,
(assignvariableop_30_adam_dense_51_bias_m/
+assignvariableop_31_adam_conv2d_60_kernel_v-
)assignvariableop_32_adam_conv2d_60_bias_v/
+assignvariableop_33_adam_conv2d_61_kernel_v-
)assignvariableop_34_adam_conv2d_61_bias_v.
*assignvariableop_35_adam_dense_50_kernel_v,
(assignvariableop_36_adam_dense_50_bias_v.
*assignvariableop_37_adam_dense_51_kernel_v,
(assignvariableop_38_adam_dense_51_bias_v
identity_40ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9▄
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*У
valueяB█(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesя
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesШ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_60_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_60_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_61_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_61_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Д
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_50_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ц
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_50_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_51_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_51_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8А
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Д
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11д
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13А
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ф
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17г
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19г
AssignVariableOp_19AssignVariableOp$assignvariableop_19_true_positives_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ф
AssignVariableOp_20AssignVariableOp"assignvariableop_20_true_negativesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Г
AssignVariableOp_21AssignVariableOp%assignvariableop_21_false_positives_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Г
AssignVariableOp_22AssignVariableOp%assignvariableop_22_false_negatives_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_60_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_60_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25│
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_61_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_61_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_50_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28░
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_50_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▓
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_51_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_51_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_60_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_60_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_61_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_61_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_50_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_50_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_51_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_51_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39Ф
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*│
_input_shapesА
ъ: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
├1
Р
C__inference_m_44_layer_call_and_return_conditional_losses_133611630

inputs,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource,
(conv2d_61_conv2d_readvariableop_resource-
)conv2d_61_biasadd_readvariableop_resource+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource
identityѕб conv2d_60/BiasAdd/ReadVariableOpбconv2d_60/Conv2D/ReadVariableOpб conv2d_61/BiasAdd/ReadVariableOpбconv2d_61/Conv2D/ReadVariableOpбdense_50/BiasAdd/ReadVariableOpбdense_50/MatMul/ReadVariableOpбdense_51/BiasAdd/ReadVariableOpбdense_51/MatMul/ReadVariableOp│
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_60/Conv2D/ReadVariableOp├
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е *
paddingVALID*
strides
2
conv2d_60/Conv2Dф
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp▒
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е 2
conv2d_60/BiasAdd
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:         Е 2
conv2d_60/ReluЈ
dropout_65/IdentityIdentityconv2d_60/Relu:activations:0*
T0*0
_output_shapes
:         Е 2
dropout_65/Identity╩
max_pooling2d_60/MaxPoolMaxPooldropout_65/Identity:output:0*/
_output_shapes
:         T *
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPool│
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_61/Conv2D/ReadVariableOpП
conv2d_61/Conv2DConv2D!max_pooling2d_60/MaxPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@*
paddingVALID*
strides
2
conv2d_61/Conv2Dф
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp░
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@2
conv2d_61/BiasAdd~
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:         I@2
conv2d_61/Reluј
dropout_66/IdentityIdentityconv2d_61/Relu:activations:0*
T0*/
_output_shapes
:         I@2
dropout_66/Identity╩
max_pooling2d_61/MaxPoolMaxPooldropout_66/Identity:output:0*/
_output_shapes
:         $@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_61/MaxPoolu
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
flatten_40/Constц
flatten_40/ReshapeReshape!max_pooling2d_61/MaxPool:output:0flatten_40/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_40/Reshapeф
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_50/MatMul/ReadVariableOpц
dense_50/MatMulMatMulflatten_40/Reshape:output:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_50/MatMulе
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_50/BiasAdd/ReadVariableOpд
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_50/BiasAddt
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_50/Reluє
dropout_67/IdentityIdentitydense_50/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_67/IdentityЕ
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes
:	ђX*
dtype02 
dense_51/MatMul/ReadVariableOpц
dense_51/MatMulMatMuldropout_67/Identity:output:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_51/MatMulД
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_51/BiasAdd/ReadVariableOpЦ
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_51/BiasAdd|
dense_51/SigmoidSigmoiddense_51/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_51/SigmoidЭ
IdentityIdentitydense_51/Sigmoid:y:0!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
Ф
▀
'__inference_signature_wrapper_133611531
conv2d_60_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallconv2d_60_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference__wrapped_model_1336111242
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
В
g
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611756

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         I@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         I@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         I@:W S
/
_output_shapes
:         I@
 
_user_specified_nameinputs
Ё
k
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_133611130

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
¤
h
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611191

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         Е 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         Е *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         Е 2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         Е 2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         Е 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Е :X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
М

р
H__inference_conv2d_61_layer_call_and_return_conditional_losses_133611221

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         I@2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         T ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         T 
 
_user_specified_nameinputs
├
J
.__inference_dropout_65_layer_call_fn_133611719

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111962
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Е :X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
╚6
ю
$__inference__wrapped_model_133611124
conv2d_60_input1
-m_44_conv2d_60_conv2d_readvariableop_resource2
.m_44_conv2d_60_biasadd_readvariableop_resource1
-m_44_conv2d_61_conv2d_readvariableop_resource2
.m_44_conv2d_61_biasadd_readvariableop_resource0
,m_44_dense_50_matmul_readvariableop_resource1
-m_44_dense_50_biasadd_readvariableop_resource0
,m_44_dense_51_matmul_readvariableop_resource1
-m_44_dense_51_biasadd_readvariableop_resource
identityѕб%m_44/conv2d_60/BiasAdd/ReadVariableOpб$m_44/conv2d_60/Conv2D/ReadVariableOpб%m_44/conv2d_61/BiasAdd/ReadVariableOpб$m_44/conv2d_61/Conv2D/ReadVariableOpб$m_44/dense_50/BiasAdd/ReadVariableOpб#m_44/dense_50/MatMul/ReadVariableOpб$m_44/dense_51/BiasAdd/ReadVariableOpб#m_44/dense_51/MatMul/ReadVariableOp┬
$m_44/conv2d_60/Conv2D/ReadVariableOpReadVariableOp-m_44_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$m_44/conv2d_60/Conv2D/ReadVariableOp█
m_44/conv2d_60/Conv2DConv2Dconv2d_60_input,m_44/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е *
paddingVALID*
strides
2
m_44/conv2d_60/Conv2D╣
%m_44/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp.m_44_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%m_44/conv2d_60/BiasAdd/ReadVariableOp┼
m_44/conv2d_60/BiasAddBiasAddm_44/conv2d_60/Conv2D:output:0-m_44/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е 2
m_44/conv2d_60/BiasAddј
m_44/conv2d_60/ReluRelum_44/conv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:         Е 2
m_44/conv2d_60/Reluъ
m_44/dropout_65/IdentityIdentity!m_44/conv2d_60/Relu:activations:0*
T0*0
_output_shapes
:         Е 2
m_44/dropout_65/Identity┘
m_44/max_pooling2d_60/MaxPoolMaxPool!m_44/dropout_65/Identity:output:0*/
_output_shapes
:         T *
ksize
*
paddingVALID*
strides
2
m_44/max_pooling2d_60/MaxPool┬
$m_44/conv2d_61/Conv2D/ReadVariableOpReadVariableOp-m_44_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$m_44/conv2d_61/Conv2D/ReadVariableOpы
m_44/conv2d_61/Conv2DConv2D&m_44/max_pooling2d_60/MaxPool:output:0,m_44/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@*
paddingVALID*
strides
2
m_44/conv2d_61/Conv2D╣
%m_44/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp.m_44_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%m_44/conv2d_61/BiasAdd/ReadVariableOp─
m_44/conv2d_61/BiasAddBiasAddm_44/conv2d_61/Conv2D:output:0-m_44/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         I@2
m_44/conv2d_61/BiasAddЇ
m_44/conv2d_61/ReluRelum_44/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:         I@2
m_44/conv2d_61/ReluЮ
m_44/dropout_66/IdentityIdentity!m_44/conv2d_61/Relu:activations:0*
T0*/
_output_shapes
:         I@2
m_44/dropout_66/Identity┘
m_44/max_pooling2d_61/MaxPoolMaxPool!m_44/dropout_66/Identity:output:0*/
_output_shapes
:         $@*
ksize
*
paddingVALID*
strides
2
m_44/max_pooling2d_61/MaxPool
m_44/flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
m_44/flatten_40/ConstИ
m_44/flatten_40/ReshapeReshape&m_44/max_pooling2d_61/MaxPool:output:0m_44/flatten_40/Const:output:0*
T0*(
_output_shapes
:         ђ2
m_44/flatten_40/Reshape╣
#m_44/dense_50/MatMul/ReadVariableOpReadVariableOp,m_44_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02%
#m_44/dense_50/MatMul/ReadVariableOpИ
m_44/dense_50/MatMulMatMul m_44/flatten_40/Reshape:output:0+m_44/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
m_44/dense_50/MatMulи
$m_44/dense_50/BiasAdd/ReadVariableOpReadVariableOp-m_44_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$m_44/dense_50/BiasAdd/ReadVariableOp║
m_44/dense_50/BiasAddBiasAddm_44/dense_50/MatMul:product:0,m_44/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
m_44/dense_50/BiasAddЃ
m_44/dense_50/ReluRelum_44/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
m_44/dense_50/ReluЋ
m_44/dropout_67/IdentityIdentity m_44/dense_50/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
m_44/dropout_67/IdentityИ
#m_44/dense_51/MatMul/ReadVariableOpReadVariableOp,m_44_dense_51_matmul_readvariableop_resource*
_output_shapes
:	ђX*
dtype02%
#m_44/dense_51/MatMul/ReadVariableOpИ
m_44/dense_51/MatMulMatMul!m_44/dropout_67/Identity:output:0+m_44/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
m_44/dense_51/MatMulХ
$m_44/dense_51/BiasAdd/ReadVariableOpReadVariableOp-m_44_dense_51_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02&
$m_44/dense_51/BiasAdd/ReadVariableOp╣
m_44/dense_51/BiasAddBiasAddm_44/dense_51/MatMul:product:0,m_44/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
m_44/dense_51/BiasAddІ
m_44/dense_51/SigmoidSigmoidm_44/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
m_44/dense_51/SigmoidЦ
IdentityIdentitym_44/dense_51/Sigmoid:y:0&^m_44/conv2d_60/BiasAdd/ReadVariableOp%^m_44/conv2d_60/Conv2D/ReadVariableOp&^m_44/conv2d_61/BiasAdd/ReadVariableOp%^m_44/conv2d_61/Conv2D/ReadVariableOp%^m_44/dense_50/BiasAdd/ReadVariableOp$^m_44/dense_50/MatMul/ReadVariableOp%^m_44/dense_51/BiasAdd/ReadVariableOp$^m_44/dense_51/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2N
%m_44/conv2d_60/BiasAdd/ReadVariableOp%m_44/conv2d_60/BiasAdd/ReadVariableOp2L
$m_44/conv2d_60/Conv2D/ReadVariableOp$m_44/conv2d_60/Conv2D/ReadVariableOp2N
%m_44/conv2d_61/BiasAdd/ReadVariableOp%m_44/conv2d_61/BiasAdd/ReadVariableOp2L
$m_44/conv2d_61/Conv2D/ReadVariableOp$m_44/conv2d_61/Conv2D/ReadVariableOp2L
$m_44/dense_50/BiasAdd/ReadVariableOp$m_44/dense_50/BiasAdd/ReadVariableOp2J
#m_44/dense_50/MatMul/ReadVariableOp#m_44/dense_50/MatMul/ReadVariableOp2L
$m_44/dense_51/BiasAdd/ReadVariableOp$m_44/dense_51/BiasAdd/ReadVariableOp2J
#m_44/dense_51/MatMul/ReadVariableOp#m_44/dense_51/MatMul/ReadVariableOp:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
Щ	
Я
G__inference_dense_50_layer_call_and_return_conditional_losses_133611788

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
њ/
г
C__inference_m_44_layer_call_and_return_conditional_losses_133611367
conv2d_60_input
conv2d_60_133611174
conv2d_60_133611176
conv2d_61_133611232
conv2d_61_133611234
dense_50_133611304
dense_50_133611306
dense_51_133611361
dense_51_133611363
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб dense_50/StatefulPartitionedCallб dense_51/StatefulPartitionedCallб"dropout_65/StatefulPartitionedCallб"dropout_66/StatefulPartitionedCallб"dropout_67/StatefulPartitionedCallи
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallconv2d_60_inputconv2d_60_133611174conv2d_60_133611176*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_60_layer_call_and_return_conditional_losses_1336111632#
!conv2d_60/StatefulPartitionedCallЦ
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111912$
"dropout_65/StatefulPartitionedCallЪ
 max_pooling2d_60/PartitionedCallPartitionedCall+dropout_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_1336111302"
 max_pooling2d_60/PartitionedCallл
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0conv2d_61_133611232conv2d_61_133611234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_61_layer_call_and_return_conditional_losses_1336112212#
!conv2d_61/StatefulPartitionedCall╔
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_1336112492$
"dropout_66/StatefulPartitionedCallЪ
 max_pooling2d_61/PartitionedCallPartitionedCall+dropout_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_1336111422"
 max_pooling2d_61/PartitionedCallё
flatten_40/PartitionedCallPartitionedCall)max_pooling2d_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flatten_40_layer_call_and_return_conditional_losses_1336112742
flatten_40/PartitionedCallЙ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_50_133611304dense_50_133611306*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_50_layer_call_and_return_conditional_losses_1336112932"
 dense_50/StatefulPartitionedCall┴
"dropout_67/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0#^dropout_66/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_67_layer_call_and_return_conditional_losses_1336113212$
"dropout_67/StatefulPartitionedCall┼
 dense_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_67/StatefulPartitionedCall:output:0dense_51_133611361dense_51_133611363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dense_51_layer_call_and_return_conditional_losses_1336113502"
 dense_51/StatefulPartitionedCallЩ
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2H
"dropout_67/StatefulPartitionedCall"dropout_67/StatefulPartitionedCall:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
┘

р
H__inference_conv2d_60_layer_call_and_return_conditional_losses_133611163

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Е 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Е 2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         └
 
_user_specified_nameinputs
╦
Я
(__inference_m_44_layer_call_fn_133611449
conv2d_60_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallconv2d_60_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_m_44_layer_call_and_return_conditional_losses_1336114302
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:         └::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:         └
)
_user_specified_nameconv2d_60_input
ѕ
ѓ
-__inference_conv2d_61_layer_call_fn_133611739

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_conv2d_61_layer_call_and_return_conditional_losses_1336112212
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         T ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         T 
 
_user_specified_nameinputs
▒
J
.__inference_flatten_40_layer_call_fn_133611777

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_flatten_40_layer_call_and_return_conditional_losses_1336112742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         $@:W S
/
_output_shapes
:         $@
 
_user_specified_nameinputs
¤
g
.__inference_dropout_65_layer_call_fn_133611714

inputs
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Е * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_1336111912
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         Е 2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Е 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Е 
 
_user_specified_nameinputs
┴
e
I__inference_flatten_40_layer_call_and_return_conditional_losses_133611772

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         $@:W S
/
_output_shapes
:         $@
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*─
serving_default░
T
conv2d_60_inputA
!serving_default_conv2d_60_input:0         └<
dense_510
StatefulPartitionedCall:0         Xtensorflow/serving/predict:ич
мg
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
Ц__call__
+д&call_and_return_all_conditional_losses
Д_default_save_signature"џd
_tf_keras_sequentialчc{"class_name": "Sequential", "name": "m_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "m_44", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_60_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_61", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "m_44", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_60_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_61", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0006000000284984708, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
э


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
е__call__
+Е&call_and_return_all_conditional_losses"л	
_tf_keras_layerХ	{"class_name": "Conv2D", "name": "conv2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}}
ж
	variables
trainable_variables
regularization_losses
	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ѓ
	variables
trainable_variables
regularization_losses
	keras_api
г__call__
+Г&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
э	

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
«__call__
+»&call_and_return_all_conditional_losses"л
_tf_keras_layerХ{"class_name": "Conv2D", "name": "conv2d_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 3, 32]}}
ж
%	variables
&trainable_variables
'regularization_losses
(	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_66", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ѓ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_61", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ж
-	variables
.trainable_variables
/regularization_losses
0	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Flatten", "name": "flatten_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
щ

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
ж
7	variables
8trainable_variables
9regularization_losses
:	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_67", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
щ

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
з
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemЋmќmЌ mў1mЎ2mџ;mЏ<mюvЮvъvЪ vа1vА2vб;vБ<vц"
	optimizer
X
0
1
2
 3
14
25
;6
<7"
trackable_list_wrapper
X
0
1
2
 3
14
25
;6
<7"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
Fmetrics

Glayers
trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
	variables
regularization_losses
Ц__call__
Д_default_save_signature
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
-
╝serving_default"
signature_map
*:( 2conv2d_60/kernel
: 2conv2d_60/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
Kmetrics
	variables

Llayers
trainable_variables
Mlayer_regularization_losses
Nnon_trainable_variables
Olayer_metrics
regularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Pmetrics
	variables

Qlayers
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics
regularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Umetrics
	variables

Vlayers
trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
Ylayer_metrics
regularization_losses
г__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_61/kernel
:@2conv2d_61/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
Zmetrics
!	variables

[layers
"trainable_variables
\layer_regularization_losses
]non_trainable_variables
^layer_metrics
#regularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
_metrics
%	variables

`layers
&trainable_variables
alayer_regularization_losses
bnon_trainable_variables
clayer_metrics
'regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
dmetrics
)	variables

elayers
*trainable_variables
flayer_regularization_losses
gnon_trainable_variables
hlayer_metrics
+regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
imetrics
-	variables

jlayers
.trainable_variables
klayer_regularization_losses
lnon_trainable_variables
mlayer_metrics
/regularization_losses
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
#:!
ђђ2dense_50/kernel
:ђ2dense_50/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
░
nmetrics
3	variables

olayers
4trainable_variables
player_regularization_losses
qnon_trainable_variables
rlayer_metrics
5regularization_losses
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
smetrics
7	variables

tlayers
8trainable_variables
ulayer_regularization_losses
vnon_trainable_variables
wlayer_metrics
9regularization_losses
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
": 	ђX2dense_51/kernel
:X2dense_51/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
xmetrics
=	variables

ylayers
>trainable_variables
zlayer_regularization_losses
{non_trainable_variables
|layer_metrics
?regularization_losses
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
=
}0
~1
2
ђ3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┐

Ђtotal

ѓcount
Ѓ	variables
ё	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
е
Ё
thresholds
єtrue_positives
Єfalse_positives
ѕ	variables
Ѕ	keras_api"╔
_tf_keras_metric«{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
Ъ
і
thresholds
Іtrue_positives
їfalse_negatives
Ї	variables
ј	keras_api"└
_tf_keras_metricЦ{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
х"
Јtrue_positives
љtrue_negatives
Љfalse_positives
њfalse_negatives
Њ	variables
ћ	keras_api"╝!
_tf_keras_metricА!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
0
Ђ0
ѓ1"
trackable_list_wrapper
.
Ѓ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
є0
Є1"
trackable_list_wrapper
.
ѕ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
І0
ї1"
trackable_list_wrapper
.
Ї	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
@
Ј0
љ1
Љ2
њ3"
trackable_list_wrapper
.
Њ	variables"
_generic_user_object
/:- 2Adam/conv2d_60/kernel/m
!: 2Adam/conv2d_60/bias/m
/:- @2Adam/conv2d_61/kernel/m
!:@2Adam/conv2d_61/bias/m
(:&
ђђ2Adam/dense_50/kernel/m
!:ђ2Adam/dense_50/bias/m
':%	ђX2Adam/dense_51/kernel/m
 :X2Adam/dense_51/bias/m
/:- 2Adam/conv2d_60/kernel/v
!: 2Adam/conv2d_60/bias/v
/:- @2Adam/conv2d_61/kernel/v
!:@2Adam/conv2d_61/bias/v
(:&
ђђ2Adam/dense_50/kernel/v
!:ђ2Adam/dense_50/bias/v
':%	ђX2Adam/dense_51/kernel/v
 :X2Adam/dense_51/bias/v
Ь2в
(__inference_m_44_layer_call_fn_133611500
(__inference_m_44_layer_call_fn_133611672
(__inference_m_44_layer_call_fn_133611651
(__inference_m_44_layer_call_fn_133611449└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
C__inference_m_44_layer_call_and_return_conditional_losses_133611630
C__inference_m_44_layer_call_and_return_conditional_losses_133611591
C__inference_m_44_layer_call_and_return_conditional_losses_133611367
C__inference_m_44_layer_call_and_return_conditional_losses_133611397└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
$__inference__wrapped_model_133611124К
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/
conv2d_60_input         └
О2н
-__inference_conv2d_60_layer_call_fn_133611692б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_conv2d_60_layer_call_and_return_conditional_losses_133611683б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_dropout_65_layer_call_fn_133611719
.__inference_dropout_65_layer_call_fn_133611714┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611709
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611704┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
4__inference_max_pooling2d_60_layer_call_fn_133611136Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
и2┤
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_133611130Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
О2н
-__inference_conv2d_61_layer_call_fn_133611739б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_conv2d_61_layer_call_and_return_conditional_losses_133611730б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_dropout_66_layer_call_fn_133611761
.__inference_dropout_66_layer_call_fn_133611766┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611756
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611751┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
4__inference_max_pooling2d_61_layer_call_fn_133611148Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
и2┤
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_133611142Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
п2Н
.__inference_flatten_40_layer_call_fn_133611777б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_flatten_40_layer_call_and_return_conditional_losses_133611772б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
о2М
,__inference_dense_50_layer_call_fn_133611797б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_dense_50_layer_call_and_return_conditional_losses_133611788б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_dropout_67_layer_call_fn_133611819
.__inference_dropout_67_layer_call_fn_133611824┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611809
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611814┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
о2М
,__inference_dense_51_layer_call_fn_133611844б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_dense_51_layer_call_and_return_conditional_losses_133611835б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
оBМ
'__inference_signature_wrapper_133611531conv2d_60_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ф
$__inference__wrapped_model_133611124ѓ 12;<Aб>
7б4
2і/
conv2d_60_input         └
ф "3ф0
.
dense_51"і
dense_51         X║
H__inference_conv2d_60_layer_call_and_return_conditional_losses_133611683n8б5
.б+
)і&
inputs         └
ф ".б+
$і!
0         Е 
џ њ
-__inference_conv2d_60_layer_call_fn_133611692a8б5
.б+
)і&
inputs         └
ф "!і         Е И
H__inference_conv2d_61_layer_call_and_return_conditional_losses_133611730l 7б4
-б*
(і%
inputs         T 
ф "-б*
#і 
0         I@
џ љ
-__inference_conv2d_61_layer_call_fn_133611739_ 7б4
-б*
(і%
inputs         T 
ф " і         I@Е
G__inference_dense_50_layer_call_and_return_conditional_losses_133611788^120б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ Ђ
,__inference_dense_50_layer_call_fn_133611797Q120б-
&б#
!і
inputs         ђ
ф "і         ђе
G__inference_dense_51_layer_call_and_return_conditional_losses_133611835];<0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         X
џ ђ
,__inference_dense_51_layer_call_fn_133611844P;<0б-
&б#
!і
inputs         ђ
ф "і         X╗
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611704n<б9
2б/
)і&
inputs         Е 
p
ф ".б+
$і!
0         Е 
џ ╗
I__inference_dropout_65_layer_call_and_return_conditional_losses_133611709n<б9
2б/
)і&
inputs         Е 
p 
ф ".б+
$і!
0         Е 
џ Њ
.__inference_dropout_65_layer_call_fn_133611714a<б9
2б/
)і&
inputs         Е 
p
ф "!і         Е Њ
.__inference_dropout_65_layer_call_fn_133611719a<б9
2б/
)і&
inputs         Е 
p 
ф "!і         Е ╣
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611751l;б8
1б.
(і%
inputs         I@
p
ф "-б*
#і 
0         I@
џ ╣
I__inference_dropout_66_layer_call_and_return_conditional_losses_133611756l;б8
1б.
(і%
inputs         I@
p 
ф "-б*
#і 
0         I@
џ Љ
.__inference_dropout_66_layer_call_fn_133611761_;б8
1б.
(і%
inputs         I@
p
ф " і         I@Љ
.__inference_dropout_66_layer_call_fn_133611766_;б8
1б.
(і%
inputs         I@
p 
ф " і         I@Ф
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611809^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ Ф
I__inference_dropout_67_layer_call_and_return_conditional_losses_133611814^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ Ѓ
.__inference_dropout_67_layer_call_fn_133611819Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђЃ
.__inference_dropout_67_layer_call_fn_133611824Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ«
I__inference_flatten_40_layer_call_and_return_conditional_losses_133611772a7б4
-б*
(і%
inputs         $@
ф "&б#
і
0         ђ
џ є
.__inference_flatten_40_layer_call_fn_133611777T7б4
-б*
(і%
inputs         $@
ф "і         ђ├
C__inference_m_44_layer_call_and_return_conditional_losses_133611367| 12;<IбF
?б<
2і/
conv2d_60_input         └
p

 
ф "%б"
і
0         X
џ ├
C__inference_m_44_layer_call_and_return_conditional_losses_133611397| 12;<IбF
?б<
2і/
conv2d_60_input         └
p 

 
ф "%б"
і
0         X
џ ║
C__inference_m_44_layer_call_and_return_conditional_losses_133611591s 12;<@б=
6б3
)і&
inputs         └
p

 
ф "%б"
і
0         X
џ ║
C__inference_m_44_layer_call_and_return_conditional_losses_133611630s 12;<@б=
6б3
)і&
inputs         └
p 

 
ф "%б"
і
0         X
џ Џ
(__inference_m_44_layer_call_fn_133611449o 12;<IбF
?б<
2і/
conv2d_60_input         └
p

 
ф "і         XЏ
(__inference_m_44_layer_call_fn_133611500o 12;<IбF
?б<
2і/
conv2d_60_input         └
p 

 
ф "і         Xњ
(__inference_m_44_layer_call_fn_133611651f 12;<@б=
6б3
)і&
inputs         └
p

 
ф "і         Xњ
(__inference_m_44_layer_call_fn_133611672f 12;<@б=
6б3
)і&
inputs         └
p 

 
ф "і         XЫ
O__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_133611130ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╩
4__inference_max_pooling2d_60_layer_call_fn_133611136ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ы
O__inference_max_pooling2d_61_layer_call_and_return_conditional_losses_133611142ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╩
4__inference_max_pooling2d_61_layer_call_fn_133611148ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ┴
'__inference_signature_wrapper_133611531Ћ 12;<TбQ
б 
JфG
E
conv2d_60_input2і/
conv2d_60_input         └"3ф0
.
dense_51"і
dense_51         X