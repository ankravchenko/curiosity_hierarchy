import sys
from PIL import Image, ImageDraw
body = Image.open('body.png', 'r')
body_mask = Image.open('body_mask.png', 'r')
head=body.resize((100,100))
head_mask=body_mask.resize((100,100))
limb = Image.open('limb.png', 'r')
limb_mask = Image.open('limb_mask.png', 'r')

ears_l, neck_l, legs_l, tail_l = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

limb_w, limb_h = limb.size

limb=limb.resize((limb_w, limb_h//2))
limb_mask=limb_mask.resize((limb_w, limb_h//2))
limb_w, limb_h = limb.size


ear=limb.resize((limb_w,limb_h*ears_l))
ear_mask=limb_mask.resize((limb_w,limb_h*ears_l))
leg=limb.resize((limb_w,limb_h*legs_l))
leg_mask=limb_mask.resize((limb_w,limb_h*legs_l))
tail=limb.resize((limb_w,limb_h*tail_l))
tail_mask=limb_mask.resize((limb_w,limb_h*tail_l))

right_ear=ear.rotate(-30, expand=True)
right_ear_mask=ear_mask.rotate(-30, expand=True)
left_ear=right_ear.transpose(Image.FLIP_LEFT_RIGHT)
left_ear_mask=right_ear_mask.transpose(Image.FLIP_LEFT_RIGHT)

tail=tail.transpose(Image.ROTATE_90)
tail_mask = tail_mask.transpose(Image.ROTATE_90)

front_leg=leg.transpose(Image.FLIP_TOP_BOTTOM).rotate(20, expand=True)
front_leg_mask=leg_mask.transpose(Image.FLIP_TOP_BOTTOM).rotate(20, expand=True)
back_leg=front_leg.transpose(Image.FLIP_LEFT_RIGHT)
back_leg_mask=front_leg_mask.transpose(Image.FLIP_LEFT_RIGHT)


print('finished loading images')

#ear=ear.resize((100,100))
#ear_mask=ear_mask.resize((100,100))
#tail = Image.open('tail.png', 'r')

body_w, body_h = body.size
head_w, head_h = head.size
ear_w, ear_h = right_ear.size
tail_w, tail_h = tail.size
leg_w, leg_h = front_leg.size

background = Image.new('RGBA', (1000, 1000), (255, 255, 255, 255))
bg_w, bg_h = background.size

#draw body
offset_center_w, offset_center_h = ((bg_w - body_w) // 2, (bg_h - body_h) // 2)
offset_center=(offset_center_w, offset_center_h)
background.paste(body, offset_center)

#draw neck and head
neck=int(100+70*(neck_l**1/2)) #neck length
offset = (bg_w//2 + neck - head_w//2, bg_h//2 - neck - head_h//2)
background.paste(head, offset, mask=head_mask)
draw = ImageDraw.Draw(background) 
draw.line((bg_w//2, bg_h//2, bg_w//2 + neck, bg_h//2 - neck), fill=(0,0,0,255), width=10)


#draw ears
offset = (bg_w //2 + neck , bg_h // 2 - neck  - ear_h )
background.paste(right_ear, offset, mask=right_ear_mask)
offset = (bg_w //2 + neck - ear_w , bg_h // 2 - neck - ear_h )
background.paste(left_ear, offset, mask=left_ear_mask)

#draw legs

offset = (bg_w //2 , bg_h // 2 + body_h//2 - 10)
background.paste(front_leg, offset, mask=front_leg_mask)
offset = (bg_w //2 - leg_w, bg_h // 2 + body_h//2 - 10)
background.paste(back_leg, offset, mask=back_leg_mask)

#draw tail
offset = (bg_w//2 - tail_w - body_w//2, bg_h//2 - tail_h//2)
background.paste(tail, offset, mask=tail_mask)

background=background.resize((256,256))

background.show()
background.save('out.png')
