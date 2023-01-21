from tkinter import *
from PIL import Image, ImageTk
import torch
import config


def get_recommendations_clicked():
    try:
        user_id = int(user_input.get())
    except ValueError:
        print("Error: input is not a number")
        return

    if user_id >= 0 and user_id < amount_of_users:
        # disable mouse
        main_frame.config(cursor="wait")

        # getting recommendations
        recommendations_id, recommendations = recommender.get_recommendations(user_id)

        # enable mouse
        main_frame.config(cursor="")

        # remove image
        image.destroy()

        recommendations_main_labels.config(text="Recommended movies for user " + str(user_id) + " :")

        # destroy previous recommendations labels if exits
        for i in range(len(recommendations_labels)):
            recommendations_labels[i].destroy()

        # remove previous recommendations labels if exits
        for i in range(len(recommendations_labels)):
            recommendations_labels.pop()

        # create recommendations labels to show the recommendations to the user
        for i in range(len(recommendations)):
            recommendations_labels.append(Label(main_frame, text=str(i+1) + ": " + str(recommendations[i]), background='#252525', fg='white', font=('italic', 16)))
            recommendations_labels[i].place(relx=0.5, rely=(0.3+0.07*i), anchor=CENTER)


# define window
window = Tk()
window.title("Welcome to Movies Recommendation System")
window.geometry('600x400')
window.minsize(700, 800)
window.configure(background='#252525')
window.iconbitmap("photos/icon.ico")

# define frame
main_frame = Frame(window, width=700, height=800)
main_frame.configure(background='#252525')
main_frame.pack()

# define image
img = Image.open("photos/pop-movies2019.jpg")
resized_image = img.resize((600, 450), Image.Resampling.LANCZOS)
new_image = ImageTk.PhotoImage(resized_image)
# Create a Label Widget to display the Image
image = Label(main_frame, image=new_image)
image.place(relx=0.5, rely=0.5, anchor=CENTER)

# define label
wait_label = Label(main_frame, text="Please wait until recommender will finish loading ... ", pady=20, background='#252525', fg='#E55555', font=('italic', 18, "bold"))
wait_label.grid(column=0, row=0)
wait_label.place(relx=0.5, rely=0.1, anchor=CENTER)

# open window
window.update_idletasks()
window.update()

# load recommender class
recommender = torch.load(config.recommender_path, map_location=config.device)
amount_of_users = recommender.get_users_amount()
wait_label.destroy()

# define label
enter_user_label = Label(main_frame, text="Please enter a user ID (0 - " + str(amount_of_users - 1) + ") for getting movies recommendations", pady=20, background='#252525', fg='white', font=('italic', 15))
enter_user_label.grid(column=0, row=0)
enter_user_label.place(relx=0.5, rely=0.05, anchor=CENTER)

# define text input
user_input = Entry(main_frame, width=10, font=('Calibri Light', 12, 'bold'), justify=CENTER)
user_input.grid(column=0, row=1)
user_input.place(relx=0.5, rely=0.1, anchor=CENTER)

# define 'get recommendations' button
get_recommendations_btn = Button(main_frame, text="Get Recommendations", command=get_recommendations_clicked, font=('italic', 13, 'bold'), bg='#E50914', fg="#FFFFFF")
get_recommendations_btn.grid(column=0, row=10)
get_recommendations_btn.place(relx=0.5, rely=0.16, anchor=CENTER)

# define labels
recommendations_main_labels = Label(main_frame, text="", pady=20, background='#252525', fg='#E55555', font=('italic', 16, 'bold'))
recommendations_main_labels.place(relx=0.5, rely=0.23, anchor=CENTER)
recommendations_labels = []

# start window main loop
window.mainloop()


