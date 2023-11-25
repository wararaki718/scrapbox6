extends Control


@onready var _label := $Label

# Called when the node enters the scene tree for the first time.
func _ready():
	var _text := "Good night world!"
	get_tree().create_timer(5.0)
	_label.set_text(_text)


# Called every frame. 'delta' is the elapsed time since the previous frame.
# func _process(delta):
# 	pass
